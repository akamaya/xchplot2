# xchplot2 SYCL T3 GPU 最適化・検証記録

- 日付: 2026-11-20
- リポジトリ: `akamaya/xchplot2`
- ブランチ: `aes-optimize`
- 対象コード: `src/gpu/T3OffsetsSycl.cpp`
- プラットフォーム: Windows 11 / AdaptiveCpp SYCL (CUDA backend) / NVIDIA RTX 4070 SUPER 12 GB
- 対象ワークロード: Chia PoS2 `k=28`, `Strength=6`, `gpu0`、1 GPU worker
- 状態: 性能計測、シャード一致テスト、完成 `.gplot` のサンプルFull Proof検証が完了。新機能は明示的に有効化する実験実装。

## 1. 目的と背景

T1/T2最適化後も、T3 match + Feistel は約17.66秒/Plotで、Plot生成全体36.75秒/Plotの約48%を占めていた。旧T3単一カーネルでは、1 laneが左エントリーを担当し、可変長の右エントリー照合ループ内でAES pairing、test filter、Feistelを実行する。そのためlane間の処理量が不均等になり、ワープ発散が生じる。

T2で有効だったwork-group内候補集約（Local Compaction）と16-way banked Tezcan AESをT3に移植して、この発散とAESテーブルアクセスを改善した。

## 2. 実装

`src/gpu/T3OffsetsSycl.cpp`に、既存実装を保存したままオプトインの新経路を追加した。実装対象は次の3経路。

- `launch_t3_match_all_buckets`
- `launch_t3_match_section_pair`
- `launch_t3_match_section_pair_split`

### Local Compaction

- work-group内で各laneが該当するペア候補を列挙する。
- `sycl::exclusive_scan_over_group`で候補の格納位置を割り当てる。
- 候補を共有ローカルメモリに格納し、lane間で分担してAES pairing、test filter、Feistel暗号化、結果書き込みを行う。
- 候補保持数はwork-groupあたり最大512。超過時はwork-group全体で元の逐次候補処理にフォールバックし、候補を切り捨てない。
- パディングされたlaneもscan/barrierに参加し、work-group内同期の整合を保つ。

### Tezcan AES

- 256エントリーのAES T0を16 way複製した16 KiBテーブルを共有メモリに展開する。
- 各laneが `static_cast<uint32_t>(local_id) & 15u` で複製バンクを選び、アクセス衝突を軽減する。
- AESターゲット計算とペアリング計算の双方に適用する。
- Tezcanを無効にした場合、Compaction経路内でも従来AES実装を使える。
- T3の最終 `proof_fragment` は従来どおりFeistelで作り、sortは後段の既存処理を用いる。

### 切替スイッチ

| 環境変数 | 説明 |
|---|---|
| `POS2GPU_T3_COMPACT=1` | T3 Local Compactionを有効化 |
| `POS2GPU_T3_TEZCAN=1` | T3 Compaction内でバンク分散Tezcan AESを使用 |
| `POS2GPU_T3_TWOPHASE=0` | 従来のT3グローバル候補バッファを用いるtwo-phaseを無効化。Compaction経路を選んだ場合はtwo-phaseをバイパスする |

両新スイッチはデフォルトOFF。T1/T2の既存最適化は変更していない。

## 3. ベンチマーク設定

T1/T2も従来の最速設定を有効にした状態で比較。各比較は `--warmup 1 -n 3`（ウォームアップ1、測定3）で、出力先はNAS共有。

```powershell
$env:POS2GPU_T1_COMPACT='1'; $env:POS2GPU_T2_COMPACT='1'; $env:POS2GPU_T2_TEZCAN='1'; $env:POS2GPU_T2_TWOPHASE='0'; $env:POS2GPU_T3_TWOPHASE='0'; $env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_PHASE_TIMING='1'; .\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 3 --warmup 1 --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test
```

Tezcan OFF比較は、同じ設定で `POS2GPU_T3_TEZCAN='0'` に変更して実行した。

## 4. ベンチマーク結果

| 構成 | T3 match + Feistel | Plot定常時間 | 月間生成量(30日換算) | VRAMピーク |
|---|---:|---:|---:|---:|
| T3従来fused (T1/T2最適化済) | 約17.66秒 | 36.75秒 | 64.3 TiB | 10,474 MiB |
| T3 Local Compaction、Tezcan OFF | 約12.94秒 | 32.11秒 | 73.6 TiB | 10,474 MiB |
| **T3 Local Compaction、Tezcan ON** | **約10.02秒** | **29.10秒** | **81.2 TiB** | **10,474 MiB** |

- Local Compaction単独: 従来T3比で約26.7%のT3時間短縮。
- Compaction + Tezcan: 従来T3比で約43.3%のT3時間短縮。
- Tezcan追加効果: Compactionのみの12.94秒から10.02秒へ、約22.6%短縮。
- Plot全体: T3変更前の36.75秒から29.10秒へ、約20.8%短縮。
- 月間生成量: 64.3 → 81.2 TiB（約26.3%増）。
- 初期SYCL基準68.17秒/Plotからの累積: 約57.3%の時間短縮、約2.34倍のスループット。

### Tezcan ON（測定3 Plot）の詳細

| 項目 | 結果 |
|---|---:|
| T1 match | 7.76〜7.80秒 |
| T2 match | 9.77〜9.81秒 |
| T3 match + Feistel | 10.014 / 10.026 / 10.029秒 |
| Plot定常時間 | 29.10秒 |
| Plot間隔の範囲 | 28.98〜29.21秒 |
| Plot間隔の標準偏差 | 0.11秒 |
| Plotサイズ | 約0.934 GiB |
| VRAMピーク | 10,474 MiB（利用可能11,069 MiB、最小空き595 MiB） |

### Tezcan OFF（測定3 Plot）の詳細

| 項目 | 結果 |
|---|---:|
| T3 match + Feistel | 12.930 / 12.948 / 12.948秒 |
| Plot定常時間 | 32.11秒 |
| Plot間隔の範囲 | 32.06〜32.20秒 |
| Plot間隔の標準偏差 | 0.08秒 |
| VRAMピーク | 10,474 MiB |

### Cold startに関する注意

`--warmup 0 --keep -n 1` で完成Plotを保持したときの計測値は **34.20秒/Plot**。フェーズのGPU処理時間は約28.69秒、うちT3 match + Feistelは9.968秒だった。初回起動の準備・コンパイル・その他オーバーヘッドが含まれるため、定常速度の29.10秒/Plotと混同しないこと。

AdaptiveCppのHIP DLL 126およびLevel Zero `zeInit` 警告は、今回のNVIDIA CUDAバックエンドでの実行・検証を妨げなかった。

## 5. 正当性の検証

### T3シャード一致（PASS 6/6）

```powershell
$env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_T3_TWOPHASE='0'; .\build\windows-sycl\tools\parity\sycl_t3_phase_sharded_parity.exe
```

| k | Seed | 結果 | フラグメント数（双方同数） |
|---:|---:|---|---:|
| 18 | 7 | PASS | 261,171 |
| 18 | 31 | PASS | 262,844 |
| 20 | 7 | PASS | 1,042,332 |
| 20 | 31 | PASS | 1,050,622 |
| 22 | 7 | PASS | 4,182,067 |
| 22 | 31 | PASS | 4,201,492 |

すべて `size=1 bytes=1`。比較はT3の単一GPU経路とシャード経路の出力集合の一致であり、独立したCPU参照との比較ではない。テスト時のStrengthは2。したがってこれ単独ではk28/Strength 6を網羅しない。

### 完成PlotのFull Proof検証（PASS）

Tezcan ON・Compaction ONで合成 `.gplot` を保存し、100チャレンジについて完全なProofを検証した。

```powershell
$env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_T3_TWOPHASE='0'; .\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 1 --warmup 0 --keep --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test
```

```powershell
$p = Get-ChildItem '\\NAS-UGREEN\personal_folder\plots_test' -Filter 'bench-*.gplot' | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if ($p) { .\build\windows-sycl\tools\xchplot2\xchplot2.exe verify $p.FullName --trials 100 --full } else { Write-Warning 'Plotが見つかりません' }
```

観測結果:

```text
[verify] 100 trials, 44 with >=1 quality chain, 91 quality chains total
[verify] 91 full proofs validated
[verify] OK (full proof validation)
```

- 対象はk28 / Strength 6。
- 検証した91個の完全Proofがすべて有効。
- サンプルベースの検証であり、すべてのチャレンジ・環境・入力を数学的に網羅するものではない。
- `bench` が生成する`.gplot`は合成データであり、ファーミング用Plotではない。

## 6. 現時点の推奨設定

PowerShell（1行で実行可能）:

```powershell
$env:POS2GPU_T1_COMPACT='1'; $env:POS2GPU_T2_COMPACT='1'; $env:POS2GPU_T2_TEZCAN='1'; $env:POS2GPU_T2_TWOPHASE='0'; $env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T3_TEZCAN='1'; $env:POS2GPU_T3_TWOPHASE='0'; $env:POS2GPU_PHASE_TIMING='1'
```

## 7. 残課題・フォローアップ

- Strength 3〜7の直接CPU/GPU比較ができるT3 parityの追加・拡充。
- 低VRAMのsection pair / splitと本番相当の多様な条件での追加検証。
- 候補上限512超過時フォールバックの発生率、GPU occupancy、共有メモリ使用量の計測。
- 次の最適化はT1/T2/T3のどこに再び時間が集中しているかを測定して判断する。
- `docs/`が`.gitignore`で無視されている環境では、ドキュメントを追加するときに `git add -f` を用いる。

この記録は2026-11-20時点の実測・テスト結果を整理したものであり、コミット・Push済みであることを示すものではない。

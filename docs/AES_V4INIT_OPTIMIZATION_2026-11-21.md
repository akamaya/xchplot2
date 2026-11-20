# AES V4INIT 最適化・採用・正当性確認記録

- 記録日：2026-11-21（JST。測定は2026-11-20 UTCから継続）
- 対象：`akamaya/xchplot2` / Chia PoS2 Plot2 / Windows 11 / SYCL・AdaptiveCpp（CUDA backend）
- GPU：NVIDIA GeForce RTX 4070 SUPER 12 GB（`gpu0`）
- Plot条件：`k=28` / Strength 6 / GPUワーカー1 / NAS SMB出力
- **採用結論：T1～T3のTezcan T0複製テーブル初期化を、`sycl::vec<uint32_t,4>` によるV4INIT（128bitストア）へ統一。個別・共通のON/OFF変数は不要。**
- **検証状況：V4INIT導入後に実行した合成k28/Strength 6 `.gplot` の `verify --full --trials 100` がPASS、114/114 full proofs validated。** 同一入力のCPU/GPU Plot全バイト一致は今回未実施。

## 1. 結論と適用範囲

T0の16倍複製テーブル（`16 * 256`個の `uint32_t`、16 KiB）を、従来は32bit単位で共有メモリへ転送していた。今回のV4INITでは、同じ値を4複製ずつ128bitベクトルで書き込む。**テーブルの全4096要素、複製配置、AESラウンド計算、検索・候補選別・出力ロジックは変えない。**

対象はSYCLカーネル中の12箇所：T1 3箇所、T2 5箇所、T3 4箇所。Tezcan専用経路は常時V4INITを使用し、T2/T3の条件付き経路ではTezcan有効時のみV4INITを使う。Tezcan無効時の従来4テーブル初期化経路は維持する。CUDA-native実装とAES単体ベンチのテスト用ソースは、本番採用パッチでは変更しない。

既存設定 `POS2GPU_T1_COMPACT`、`POS2GPU_T2_COMPACT`、`POS2GPU_T3_COMPACT`、`POS2GPU_T2_TEZCAN`、`POS2GPU_T3_TEZCAN`、`POS2GPU_FINE_DIRECT`、Fine Bits、TwoPhase は従来どおり。**`POS2GPU_AES_VEC_INIT` は正式採用コードで使用しない。**

## 2. 研究の経緯

T1～T3高速化が進み、AESラウンドそのものの共有メモリ参照が残る律速となった。Nsight ComputeではRotate16（16複製）の共有メモリロードはほぼ2 wavefront/LDS命令で、メモリ命令の圧迫（MIO throttle）が目立った。AES計算自体の大変更は、追加演算や占有率悪化で逆効果になる懸念があり、まず**初期化だけ**を対象にした。

主な候補は次のとおり。

| 候補・調査 | 確認したこと | 判断 |
|---|---|---|
| 従来T0～T3方式 | AES単体512ラウンド・262,144状態で約1.609ms | 比較基準 |
| Rotate16 / 16複製T0 | 同条件で約1.038ms。計算本体では優位 | 維持 |
| Rotate32 / 32複製 | 約1.04ms。共有メモリ増加でoccupancyが低下し、実質的な改善なし | 不採用 |
| 圧縮S-box方式等 | 演算増加のコストが大きく、Rotate16より遅い候補があった | 不採用 |
| V2INIT / V4INIT | T0を64bit/128bit単位で複製初期化。AESラウンドは維持 | **V4INIT採用** |

代表的なAES単体測定（262,144状態、256ラウンドペア＝512 AESラウンド）：Rotate16約1.038ms、V2INIT約1.034ms、V4INIT約1.035ms。512ラウンドの長い計算では初期化の短縮効果は小さい。一方、32 AESラウンドでは、10回のペア測定でRotate16平均 **0.0913ms**、V4INIT平均 **0.0888ms**（約**2.74%短縮**、10組中9組でV4INITが高速）。

このAES単体実験の値は実Plotの時間ではない。実Plotでの採否は後述のT1～T3 A/B測定を優先した。

## 3. Nsight Computeによる確認（SASS）

`aes_rotate16_sass.txt` と `aes_v4init_sass.txt` を確認。V4INITは希望した128bit命令に実際にコンパイルされていた。

| 指標 | Rotate16従来版 | V4INIT |
|---|---:|---:|
| shared store SASS | `STS [R5], R0` | **`STS.128 [R0+-0x8], R8`** |
| shared store命令数（warp実行） | 131,072 | **32,768（1/4）** |
| 初期化に関するglobal load命令数（warp実行） | 131,072 | **32,768（1/4）** |
| shared store wavefront数 | 131,072 | 131,072 |
| shared store bank conflicts | 0 | 0 |

幅を広げてもshared memoryへ転送する総バイト数とwavefront数は変わらず、**命令発行数の削減**が主な効果。AESラウンド中のshared LDSは32bit参照のままである。

**注意：** SASSのRotate16プロファイルは512 AESラウンド、V4INITプロファイルは32 AESラウンド。初期化命令の命令種別・回数の比較はできるが、カーネル全体のstall sample割合を同条件の直接比較として扱わない。

## 4. 実Plotカーネルの同設定A/B結果

両条件で以下を統一：

- `POS2GPU_FINE_DIRECT=1`、`POS2GPU_T1_FINE_BITS=8`、`POS2GPU_T2_FINE_BITS=16`、`POS2GPU_T3_FINE_BITS=16`
- `POS2GPU_T1_COMPACT=1`、`POS2GPU_T2_COMPACT=1`、`POS2GPU_T3_COMPACT=1`
- `POS2GPU_T2_TEZCAN=1`、`POS2GPU_T3_TEZCAN=1`、`POS2GPU_T2_TWOPHASE=0`、`POS2GPU_T3_TWOPHASE=0`
- 候補容量は T1=512、T2=480、T3=330。`POS2GPU_MATCH_DIAG=0`、`POS2GPU_PHASE_TIMING=1`
- コマンド：`bench -k 28 -s 6 -n 1 --warmup 1 --devices gpu0 -o <NAS出力先>`

同じ設定で**OFF→ONを2組**測定し、ウォームアップ後の各1 Plotの`[phase-timing]`を平均した。試験パッチの `POS2GPU_AES_VEC_INIT=0/1` のみ切り替えた。正式採用後はこの切り替えは存在しない。

| 処理 | OFF：32bit初期化 | ON：V4INIT | 短縮量 | 時間短縮率 |
|---|---:|---:|---:|---:|
| T1 match | 7,802.9 ms | 7,826.3 ms | -23.4 ms | **-0.30%** |
| T2 match | 8,685.3 ms | **8,505.9 ms** | **179.4 ms** | **2.07%** |
| T3 match + Feistel | 9,066.3 ms | **8,809.1 ms** | **257.2 ms** | **2.84%** |
| **Phase合計** | **26,916.0 ms** | **26,450.4 ms** | **465.6 ms** | **1.73%** |

同条件の個別Phase合計ログは、OFF **26,941.3 / 26,890.7 ms**、ON **26,452.7 / 26,448.1 ms**。T2/T3は2組とも改善。T1は明確な改善がなく、わずかに悪化したが、スイッチ増加による保守負荷を避けるため、ユーザー判断によりT1も含めてV4INITを採用。

`[bench] steady-state` はOFF **27.08 / 26.44 s/plot**、ON **26.97 / 26.19 s/plot**。**26.19秒は今回の最速観測値であり、10 Plot平均値ではない。** NAS書込、他処理の並列性、GPU状態の影響を含むため、V4INITの効果を評価するときは上記Phase A/B比較（1.73%）を主指標とする。

## 5. 正当性確認

### CPU側・構造テスト

- 元の32bit初期化とV4INITでshared T0全**4096要素**が一致。
- **256個のT0値 × 16バンク**の複製配置を検査。
- ランダム**256状態 × 32 AESラウンド**の128bit出力一致、およびTezcan無効時の4テーブル方式の維持を確認。
- 12か所のパッチ適用・復元ライフサイクルを模擬テストで確認。

### Windows GPUで生成した実ファイルの検証

正式採用版で、以下を実行。

```powershell
.\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 1 --warmup 0 --keep --devices gpu0 -o <Plot出力先>
```

生成された合成 `.gplot` は **0.934 GiB**。対象はこの実行で保持した合成 `.gplot` 1個。

```powershell
.\build\windows-sycl\tools\xchplot2\xchplot2.exe verify <生成ファイル.gplot> --full --trials 100
```

出力ログ：

```text
[verify] 100 trials, 46 with >=1 quality chain, 114 quality chains total
[verify] 114 full proofs validated
[verify] OK (full proof validation)
```

**判定：100チャレンジから見つかった114個のフルプルーフをすべて正常に検証。** この生成Plotについて、AES初期化変更を含む最適化後のPlot生成処理が機能したことを確認できた。

**限界：** `verify --full` は試行したチャレンジから見つかるProofを検証するサンプリングであり、全バイトを網羅しない。今回V4INIT導入後の**同一入力CPU/GPU Plot全バイト比較は行っていない**。また `bench` の合成Plotは**非農業用**である。`--warmup 0` の初回実行にはJITコンパイル警告があり、33.62秒という値は速度評価から除外する。

## 6. 先行最適化との関係（README一覧表の追記分）

### Fine Bucket：検索段階の改善

**10 Plot平均、1 Plotウォームアップ除外**。以下はT2/T3の二分探索・16+4bit検索を比較した以前の測定で、今回の1 Plot A/Bとはサンプル数が異なる。

| Fine Bits（T1/T2/T3）と検索 | T2 match_fused | T3 match_fused | Phase合計 | Steady-state |
|---|---:|---:|---:|---:|
| 8/8/8（二分探索） | 9,821.1 ms | 10,040.4 ms | 28,907.9 ms | 28.74 s/plot |
| 8/14/14（二分探索） | 9,448.9 ms | 9,622.8 ms | 28,133.1 ms | 28.16 s/plot |
| 8/16/16（二分探索） | 9,273.1 ms | 9,455.6 ms | 27,809.7 ms | 27.95 s/plot |
| **8/16/16（16+4bit直接参照）** | **9,094.9 ms** | **9,248.4 ms** | **27,404.5 ms** | **27.27 s/plot** |

16+4bit直接参照は、16bit二分探索比でSteady-state **0.68秒（2.43%の時間短縮）**。8bit基準比では1.47秒（5.11%短縮）。索引は約384 MiBの共用scratch、255件以上のグループは二分探索にフォールバック。詳細は `docs/FINE_BUCKET_OPTIMIZATION_2026-11-21.md` を参照。

### T2/T3のLocal Compaction候補容量

別測定では、T1候補上限512を維持し、T2を512→480、T3を512→330に縮小。`match_fused`代表値で、T2 **9,080→8,767 ms（-313ms、3.44%短縮）**、T3 **9,225→8,932 ms（-293ms、3.18%短縮）**。合計約**0.61秒/Plot**のカーネル短縮。ただし、これはその条件下のカーネル測定であり、**新しいSteady-stateの独立した平均値はない**。T3上限330超過時には候補を破棄せずフォールバックする。詳細は `docs/GPU_PAIR_CAP_T1_T3_2026-11-21.md` を参照。

先行表とV4INITの表は測定回数・条件が異なる。`68.17→26.19 s/plot` は観測値同士の算術比較として**約61.6%短縮、約2.60倍のPlot数/時**だが、同条件・同回数の厳密A/B実験を意味しない。

## 7. 運用・Git管理

採用版はON/OFF変数 `POS2GPU_AES_VEC_INIT` を参照しない。環境変数 `POS2GPU_AES_BENCH_ONLY=INIT` はベンチ用であり本番Plotパイプラインを切り替えない。ローカルのREADME更新後に `git diff -- README.md`、`git diff --check` を確認する。採用コード（T1/T2/T3）と、調査用ベンチソース `tools/parity/sycl_aes_bench.cpp` の巨大な実験差分は混ぜない。Windowsテスト用ソースのコンパイル修正も別コミットに切り分ける。

## 8. 参照資料

- 本リポジトリ: `docs/OPTIMIZATION_HISTORY_2026-11-20.md`（初期T1調査）
- `docs/FINE_BUCKET_OPTIMIZATION_2026-11-21.md`（Fine Bucket検索）
- `docs/GPU_PAIR_CAP_T1_T3_2026-11-21.md`（候補容量とNsight）
- ローカル解析資料: `aes_rotate16_ncu_analysis_20261121.md`、`aes_rotate16_sass.txt`、`aes_v4init_sass.txt`
- 今回の実機ログ: 会話に提示されたOFF/ONベンチ2組、および`verify --full`結果

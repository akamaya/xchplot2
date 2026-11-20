# T1–T3 GPU候補バッファ容量の最適化とNsight Compute調査

- 記録日：2026-11-21（日本時間、検証は2026-11-20 UTCから継続）
- 対象：`xchplot2` / Windows 11 / AdaptiveCpp SYCL（CUDA backend）
- GPU：GeForce RTX 4070 SUPER（12 GB、Compute Capability 8.9、56 SM）
- 条件：`bench -k 28 -s 6 -n 1 --warmup 1 --devices gpu0`、計測は主として診断OFF
- **最終採用設定：T1 `pair_cap=512`、T2 `pair_cap=480`、T3 `pair_cap=330`**

## 1. 結論

T2とT3は、ワークグループ内の候補ペア用共有メモリを縮小すると、共有メモリ制約によるSM常駐ブロック上限が理論上3→4に増える境界を越え、各約3%高速化した。最終設定でNsight Computeを実行した結果、いずれも **共有メモリによる上限が4ブロック/SM、理論占有率66.67%** であることを確認した。

T1は512→352→344件と減らしても処理時間がほぼ変わらなかったため、512件を維持する。T1のボトルネック調査では、共有メモリのロードに非常に多くのBank Conflictが発生し、メモリ命令パイプラインが高負荷であると分かった。ただし、テーブル方式やAES計算そのものは今回変更しない。

## 2. 速度測定（`match_fused`）

`-n 1 --warmup 1`を複数回実行。**JITが発生したウォームアップPlotは除外**し、測定Plotの`[fine-timing] ... match_fused`を比較した。差が小さいものは測定ばらつきとして扱う。

| 段階 | 候補容量 | match_fused（代表値） | 比較対象 | 差 | 判断 |
|---|---:|---:|---:|---:|---|
| T1 | 512 | 約7,794 ms | — | — | 維持 |
| T1 | 352 | 約7,786 ms | T1/512 | -0.1% | 明確な改善なし |
| T1 | 344 | 約7,788 ms | T1/512 | -0.1% | 明確な改善なし |
| T2 | 512 | 約9,080 ms | — | — | 旧設定 |
| **T2** | **480** | **約8,767 ms** | T2/512 | **-313 ms（-3.44%）** | **採用** |
| T3 | 512 | 約9,225 ms | — | — | 旧設定 |
| T3 | 320 | 約8,928 ms | T3/512 | -3.2% | 速いがフォールバック多め |
| **T3** | **330** | **約8,932 ms** | T3/512 | **-293 ms（-3.18%）** | **採用** |
| T3 | 336 | 約9,234 ms | T3/330 | +3.4% | 共有メモリ境界と整合する性能低下 |

T2＋T3の短縮は合計**約0.61秒/Plot**（各カーネル時間の足し算）。プロファイル中の`steady-state`や`end-to-end`は測定オーバーヘッドや非同期書き込みの影響があり、この表の比較には用いない。

## 3. 候補超過と正確性

`POS2GPU_MATCH_DIAG=1`、`POS2GPU_MATCH_DIAG_STRIDE=128` で確認した。

| 段階・容量 | フォールバック回数（各Plot、EXACT） | 補足 |
|---|---|---|
| T1/352 | 0、0 | サンプル最大候補数316、323 |
| **T2/480** | **0、0** | サンプル最大候補数342、331 |
| **T3/330** | **259、270、279、290** | 約6,816万ワークグループ/Plot中の発生回数 |

ログ項目名の`fallback_over_512_groups(EXACT)`は旧ラベルのままだが、**比較する上限はソース内の実際の`pair_cap`**。`sampled_groups`/`max_group_candidates(sample)`は抽出計測、`fallback ... (EXACT)`は全グループでの計測を意味する。

T3で330件を超えた場合は、グループ一様のフォールバック処理で**全候補を再走査して処理する**。候補を黙って破棄する設計ではない。発生件数だけで全ケースの正確性を証明するわけではないため、実プロットの検証は引き続き別途必要。

## 4. Nsight ComputeでのSM占有率（最終設定）

Nsight Compute CLI 2025.2.1、`LaunchStats`＋`Occupancy`で取得。`--launch-count 1`、診断OFF、対象の`launch_t[123]_match_all_buckets_compact`をそれぞれ指定した。

| 指標 | T1 / 512 | T2 / 480 | T3 / 330 |
|---|---:|---:|---:|
| スレッド/ブロック | 256 | 256 | 256 |
| Registers Per Thread | 40 | 40 | 40 |
| Dynamic Shared Memory / Block | 20,484 B | 24,068 B | 24,308 B |
| Static Shared Memory / Block | 128 B | 128 B | 128 B |
| Driver Shared Memory / Block | 1,024 B | 1,024 B | 1,024 B |
| Block Limit Registers | 6 | 6 | 6 |
| **Block Limit Shared Mem** | **4** | **4** | **4** |
| Block Limit Warps | 6 | 6 | 6 |
| Theoretical Occupancy | 66.67% | 66.67% | 66.67% |
| **Achieved Occupancy** | **62.86%** | **63.11%** | **62.74%** |

SM当たり共有メモリ容量は102,400 B（100 KiB）。256 B単位の割当丸めと、静的128 B＋ドライバ1,024 Bを同様に仮定すると：

- **T2/480**：`ceil256(24068+128+1024)=25344 B`、4ブロックで101,376 B → 収容可能。
- **T2/512（推定）**：`ceil256(24580+128+1024)=25856 B`、4ブロックで103,424 B → 4ブロック不可。
- **T3/330**：`ceil256(24308+128+1024)=25600 B`、4ブロックで102,400 B → ちょうど収容可能。
- **T3/336（推定）**：`ceil256(24452+128+1024)=25856 B`、4ブロックで103,424 B → 4ブロック不可。
- **T1/512**：`ceil256(20484+128+1024)=21760 B` → 共有メモリ制約で4ブロック。
- **T1/352・344（推定）**：それぞれ約20,480 B/ブロック → 5ブロックに収まる計算。ただしこの2設定でのNsight実測は未取得であり、実行時間には改善が見られなかった。

**確定しているのは最終構成での4ブロック上限。旧設定の3ブロック・T1縮小時の5ブロックはリソース量からの推定であり、直接測定値ではない。**

## 5. T1のボトルネック調査

T1/512の`SpeedOfLight`、`SchedulerStats`、`WarpStateStats`、`MemoryWorkloadAnalysis`および共有メモリカウンタを取得した。

| 指標 | 実測値 |
|---|---:|
| L1/TEX Cache Throughput / Mem Pipes Busy | 99.21% |
| DRAM Throughput | 2.88% |
| L2 Cache Throughput | 3.90% |
| L1/TEX Hit Rate | 24.51% |
| L2 Hit Rate | 74.51% |
| Issued Warp Per Scheduler | 0.66 |
| No Eligible | 33.82% |
| MIO命令キュー満杯による待機 | 発行命令間隔11.4サイクル中、平均3.9サイクル（約34.1%） |

共有メモリの命令と競合：

| 指標 | Load | Store |
|---|---:|---:|
| SASS shared-memory命令数 | 566,499,736,410 | 903,504,334 |
| Bank Conflicts | 547,538,649,451 | 48,594,956 |
| Wavefronts | 1,114,200,944,092 | 953,779,703 |
| Wavefronts / 命令 | **約1.967** | 約1.056 |

T1の主要な共有メモリ参照は`T1OffsetsSycl.cpp`の`tezcan_round()`からのAES T-tableロードである。16複製のテーブルを`AES byte index × 16 + (local_id & 15)`で参照するため、32 laneのワープ内でテーブル複製番号が重なり、Bank Conflictを生みやすい。実測は**ロード当たりほぼ2 Wavefront**であり、この仮説と整合する。

一方、MIO待機には共有メモリ以外の要因も含まれ、プロファイラの各比率は**AES共有メモリだけの寄与を厳密に分離した数値ではない**。Bank Conflictが半減してもカーネル実行時間が半減するわけではない。

## 6. T1をこれ以上変更しない理由

- 16→32複製はAESテーブルだけで**16 KiB→32 KiB**となる。T1候補容量を512→352にしても節約は1,280 Bにすぎず、常駐ブロック数の低下が懸念される。
- S-box圧縮などの方式変更はテーブルロードを減らせても**AES再構成の演算が増える**ため、今回の方針では実施しない。
- 同じT-table方式では、通常のAESラウンドに必要なランダム参照16回を大きく削減する余地が小さい。初回・最終ラウンドの微小な削減はコンパイラが既に行っている可能性がある。
- T1候補容量削減（352・344）はベンチで有意差がなかった。改善効果が未確認の変更を追加しない。

**今回の最適化はT1=512、T2=480、T3=330で完了。** 将来改善する場合は、現設定を基準に別ブランチで検証する。

## 7. 再現用コマンド（PowerShell、1行ずつ）

ビルド（16並列）：

```powershell
cmake --build .\build\windows-sycl --config Release --target xchplot2 -j 16
```

測定条件（新しいPowerShellセッションでは再設定が必要）：

```powershell
$env:POS2GPU_FINE_DIRECT="1"; $env:POS2GPU_T1_COMPACT="1"; $env:POS2GPU_T1_FINE_BITS="8"; $env:POS2GPU_T2_COMPACT="1"; $env:POS2GPU_T2_TEZCAN="1"; $env:POS2GPU_T2_TWOPHASE="0"; $env:POS2GPU_T2_FINE_BITS="16"; $env:POS2GPU_T3_COMPACT="1"; $env:POS2GPU_T3_TEZCAN="1"; $env:POS2GPU_T3_TWOPHASE="0"; $env:POS2GPU_T3_FINE_BITS="16"; $env:POS2GPU_MATCH_DIAG="0"; $env:POS2GPU_PHASE_TIMING="1"
```

```powershell
.\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 1 --warmup 1 --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test
```

重要：**`pair_cap`はC++ソースのコンパイル時定数**であり、上記環境変数で変更されない。コミット時にローカルソースのT1/T2/T3の値を確認すること。GitHubの既存デフォルトブランチのコードとローカル作業ツリーは一致しているとは限らない。

プロファイルには管理者権限でのPowerShell起動が必要な場合がある（`ERR_NVGPUCTRPERM`）。複数パスのNsight測定はGPUカーネルをリプレイするため、**そのときのPlot所要時間を通常ベンチと比較しない**。

## 8. コミット対象と留意点

- `src/gpu/T1OffsetsSycl.cpp`：最終T1容量 **512**（変更なしならステージ不要）。
- `src/gpu/T2OffsetsSycl.cpp`：最終T2容量 **480**（ローカルの対象箇所を確認）。
- `src/gpu/T3OffsetsSycl.cpp`：最終T3容量 **330**（ローカルのcompact処理箇所を確認）。
- `docs/GPU_PAIR_CAP_T1_T3_2026-11-21.md`：本メモ。

`git diff --cached`で意図しない変更やベンチログを巻き込んでいないことを確認してからコミットする。

# Chia Plot2 / xchplot2 GPU高速化 調査・実装ログ

> **記録日:** 2026-11-20  
> **環境:** Windows 11 / WDDM、RTX 4070 SUPER 12GB（Plot実行）、RTX 5070（表示側）、xchplot2 v0.14.1ベース、SYCL / AdaptiveCpp、作業ブランチ `aes-optimize`  
> **対象:** k28 / Strength 6 / 1 GPU / warmup 1 plot + 測定3 plots  

## 要約

- **最終結果:** 1 Plotあたり **68.17秒 → 44.57秒**（初期SYCL比 **34.6%短縮**、処理速度約**1.53倍**）。
- **T1:** Tezcan AESを導入しただけでは17.005秒→16.535秒（**2.8%短縮**）。候補をGPU共有メモリ上で詰め直す**Local Compaction**を追加し、16.171秒→7.800秒（**51.8%短縮**）。
- **T2・T3:** デフォルトONの既存二段階処理で候補バッファの `OVERFLOW` が多発していた。両方OFFで**58.32秒→44.57秒**（さらに**13.75秒短縮**）。
- **最終スループット:** **53.0 TiB/月**、VRAMピーク**10,474 MiB**。

## 1. 全体のベンチマーク推移

| 段階 | 変更 | steady-state / Plot | T1 match | T2 match | T3 match | VRAMピーク |
|---|---|---:|---:|---:|---:|---:|
| ① 初期SYCL | 従来AES、二段階処理ON | **68.17秒** | 17.005秒 | 24.831秒 | 24.993秒 | ― |
| ② Tezcan移植 | T1 AESのみTezcan系に変更 | **67.84秒** | 16.535秒 | ― | ― | ― |
| ③ 後半の基準 | Tezcan ON、T1集約OFF、二段階ON | **66.71秒** | 16.171秒 | 24.434秒 | 24.562秒 | 10,526MiB |
| ④ T1集約ON | 候補のLocal Compaction | **58.32秒** | 7.800秒 | 24.447秒 | 24.569秒 | 10,526MiB |
| ⑤ 最終 | T1集約ON、T2/T3二段階OFF | **44.57秒** | 7.796秒 | 17.538秒 | 17.669秒 | 10,474MiB |

※秒/Plotはベンチマークが出力したsteady-state値。T1〜T3はフェーズ計測値。両者の和が完全一致するわけではない。最初の2点は前の計測条件、③～⑤は比較的近い条件での比較。

```mermaid
xychart-beta
    title "1 Plot 生成時間（小さいほど速い）"
    x-axis [初期, Tezcan, 集約OFF, 集約ON, 二段階OFF]
    y-axis "秒" 0 --> 75
    bar [68.17, 67.84, 66.71, 58.32, 44.57]
```

## 2. WindowsでのVRAM不足

RTX 4070 SUPERは公称12GBだが、WindowsのWDDM稼働中はドライバー・デスクトップ・別プロセス等がVRAMを使う。

- `nvidia-smi` のVRAM容量：約**12,282MiB**。
- ベンチマーク時の利用可能容量：約**11,069MiB**（**10.81GiB**）。
- 当初のCUDA一括バッファ（`d_storage`、`pair_a`、`pair_b`等）の要求は概算**11,745MiB（11.47GiB）**で、利用可能容量を超過。CUDA側ではストリーミング処理に移行。
- SYCL経路ではバッファを別ステージと重ねて再利用するなどして、実測ピーク**10,526MiB**で実行できた。CUDA一括確保と比べて約1GiB規模を削減。

ここで不足していたのは主に**GPU VRAM**であり、PC主記憶128GBではない。CUDA側もストリーミング処理で生成自体は可能。

## 3. 既存のGPU AES高速化方式（Tezcan系）

PoS2のAES系ハッシュは2種類のラウンドキーを交互に使う。Strength 6では `16 << 4 = 256` 回のペアを回すため**512 AESラウンド**。

- 従来：4個のAES T-table（T0〜T3）を使用。shared memory **4KiB/ブロック**。
- Tezcan系：T0のみを読み、T1〜T3をバイト回転で計算。**T0をレーン別に複製**することでバンク競合を削減。16複製では**16KiB/ブロック**。
- CUDAは `__byte_perm()` を使用。SYCL実装ではビット回転を使用。

**512 AESラウンド・4,194,304 states**

| 環境 | 従来T-table | Tezcan 16x | 倍率 |
|---|---:|---:|---:|
| CUDA/NVCC | 24.371ms | 14.085ms | **1.73倍** |
| SYCL/AdaptiveCpp | 21.270ms | 13.595ms | **1.56倍** |

**SYCLラウンド別：** 32 AES rounds 1.587→1.079ms（1.471倍）、128 rounds 6.136→3.469ms（1.769倍）、512 rounds 21.270→13.595ms（1.564倍）。各条件で結果不一致0件。

ただしCUDAとSYCLの絶対値は計測時点が異なるので、各方式の従来比を重視する。

## 4. T1への組み込みで改善しなかった

AES単体で1.56倍速くなったにもかかわらず、実T1は17.005→16.535秒（約2.8%短縮）。

T1では各スレッドが①Target AES→②候補の二分探索→③見つかった候補の件数だけPairing AES→④条件を満たすペアの出力、を行う。候補数が不均一で、同じwarp内のスレッドが異なる回数AESを実行する**warp divergence**が疑われた。

## 5. 合成ベンチマークで原因を検証

512ラウンドで、候補を各レーンに固定で与える場合・平均1件のポアソン分布で与える場合・warp単位にグループ化した場合を比較。

| 候補パターン | 従来 | Tezcan 16x | 高速化倍率 | 候補側の推定レーン稼働率 |
|---|---:|---:|---:|---:|
| FIXED（固定） | 42.565ms | 27.956ms | **1.523倍** | 100%（モデル） |
| RANDOM（平均1件） | 65.649ms | 62.094ms | **1.057倍** | 28.39%（モデル） |
| GROUPED（warp単位集約） | 43.212ms | 28.799ms | **1.500倍** | 100%（モデル） |

RANDOMではAES単体の高速化効果がほぼ消え、GROUPEDでは回復した。バイト回転方式の移植失敗ではなく、**候補数のばらつきが主要因**と考えられる。28.39%は合成モデル値であり、性能カウンターによる実測値ではない。

## 6. T1にLocal Compactionを実装

```mermaid
flowchart LR
    A[各レーンでTarget AES] --> B[候補検索と件数計測]
    B --> C[グループ内exclusive scan]
    C --> D[共有メモリに候補ペアを詰める]
    D --> E[連続スレッドでPairing AES]
    E --> F[出力]
```

`src/gpu/T1OffsetsSycl.cpp` に追加した実験的機能。256スレッド/ブロックで候補を集め、`sycl::exclusive_scan_over_group` でオフセットを計算。候補を**最大512件**まで `pairs_l` / `pairs_r`（各u32）へ格納し、連続したレーンへ再配置する。512件超は従来のループへフォールバックし、候補を捨てない設計。

- 共有メモリ：AESテーブル16KiB＋候補ペア512×8B＝4KiB＋カウンター＝約**20KiB/ブロック**。
- ここでの共有メモリは**GPUチップ内SMのSRAM**。PCのRAMでもGPUのGDDR VRAMでもない。
- **追加VRAMほぼ0**。実測VRAMピークはOFF/ONとも**10,526MiB**。
- T1 match：**16.171秒→7.800秒（51.8%短縮、2.07倍）**。
- Plot steady-state：**66.71秒→58.32秒（12.6%短縮）**。

`POS2GPU_T1_COMPACT=1` で有効。未指定ではOFF。

## 7. T2/T3の二段階処理を見直した

T2/T3はもともと「候補検索カーネル→グローバルVRAM上の候補バッファ→Pairing AESカーネル」の二段階処理をデフォルトでONにしていた。T1局所集約と違って**別カーネル・別VRAMバッファ**を使う。

候補容量の式：

```cpp
cand_cap = (out_capacity / num_buckets_in_range) * 6 + 4096;
```

Strength 6では出力に通る割合が小さく、多くの候補を保持する必要がある。実測した二段階候補用のVRAM予算は**49MiB**。デバッグログで大量の `OVERFLOW` を確認した。

**OVERFLOW時は二段階経路の結果を巻き戻して、単一カーネルの経路で再計算する**実装。その無駄な処理がT2/T3の時間を延ばしていた。

| 指標 | 二段階ON | 二段階OFF | 改善 |
|---|---:|---:|---:|
| T1 match | 7.800秒 | 7.796秒 | ほぼ同じ |
| T2 match | 24.447秒 | **17.538秒** | −6.916秒 |
| T3 match | 24.569秒 | **17.669秒** | −6.900秒 |
| Plot steady-state | 58.32秒 | **44.57秒** | **−13.75秒（23.6%）** |
| VRAMピーク | 10,526MiB | **10,474MiB** | −52MiB |

二段階処理が常に遅いという結論ではない。今回の**Strength 6・12GB VRAM・バッファ予算**の組み合わせで不利だった。

## 8. 最終性能

- 初期SYCL 68.17秒/Plot → **44.57秒/Plot**：**34.6%短縮**。
- 後半の同条件比較 66.71秒/Plot → **44.57秒/Plot**：**33.2%短縮**。
- 35.4TiB/月 → **53.0TiB/月**（後半比較、約49.7%増）。
- VRAMピーク 10,526MiB → **10,474MiB**、実行時空き約595MiB。
- 最終T1 7.796秒、T2 17.538秒、T3 17.669秒。**今後はT2/T3が主なボトルネック**。

## 9. 再現設定・コマンド

```powershell
cd C:\dev\xchplot2
$env:POS2GPU_T1_COMPACT="1"
$env:POS2GPU_T2_TWOPHASE="0"
$env:POS2GPU_T3_TWOPHASE="0"
$env:POS2GPU_T2_TWOPHASE_DEBUG="0"
$env:POS2GPU_T3_TWOPHASE_DEBUG="0"
$env:POS2GPU_PHASE_TIMING="1"

.\build\windows-sycl\tools\xchplot2\xchplot2.exe bench `
    -k 28 -s 6 -n 3 --warmup 1 --devices gpu0 `
    -o D:\plots_test
```

**T1のCPU/GPU一致検証（今後再実施）**

```powershell
$env:POS2GPU_T1_COMPACT="1"
.\build\windows-sycl\tools\parity\sycl_t1_parity.exe --k 22 --strength 6
```

**正当性の状況：** Tezcan AESの単体ベンチは全件一致。Tezcan導入段階のT1ではCPU/GPUが **4,189,913件一致**。ただし、**Local CompactionをONにした最終コードについて、CPU/GPU一致の実行結果は会話に残っていない**。この検証と生成Plotの検証を終えるまでは、本番配布用の完成版とは扱わない。

## 10. 次の課題

1. **Local Compaction ONで正当性確認**：複数のStrength、k、Plotを検証。
2. **T2/T3へTezcan AESと局所集約を段階導入**：分岐抑制とshared memory増加によるoccupancyへの影響を計測。
3. **二段階処理の改善**：Strength依存の候補容量、早期OVERFLOW検出、再計算回避を検討。Strength 2など別条件も評価。
4. **再現性の高い比較**：同一GPU、負荷、クロック、ウォームアップ、測定回数を固定。

---

※ 再現コマンドの `D:\plots_test` は公開用の保存先例であり、実測時には別の出力先を使用。測定値・ハードウェア条件は実測ログに基づく。

**主な成果：** AES単体の1.56倍の高速化よりも、**候補を詰めてwarpを活用すること**と、**OVERFLOW後の再計算を避けること**がPlot全体の性能を大きく改善した。

## 11. 最終ベンチマークの実測ログ（抜粋）

以下はT1局所集約ON、T2/T3二段階処理OFF、warmup 1・測定3の最終ログ。フェーズ値はms。

| Plot（測定分） | T1 match | T2 match | T3 match + Feistel | Phase total |
|---|---:|---:|---:|---:|
| 1 | 7,785.9 | 17,549.6 | 17,686.5 | 44,395.2 |
| 2 | 7,799.9 | 17,548.0 | 17,645.2 | 44,308.4 |
| 3 | 7,803.1 | 17,517.5 | 17,674.9 | 44,216.0 |

```text
[batch:gpu0] vram: peak 10474 MiB of 11069 free
[bench] steady-state: 44.57 s/plot, 0.934 GiB/plot (k=28, 1 worker)
[bench] per-plot spread: interval 44.57 s (min 44.26 / max 44.83, σ 0.29)
[bench] end-to-end: 0.0736 TiB/hour | 1.77 TiB/day | 53 TiB/month (30d)
```

**注記:** `steady-state` は実際のPlot完成間隔、`phase total` はGPU処理のフェーズ計測であり、I/O等により厳密には一致しない。

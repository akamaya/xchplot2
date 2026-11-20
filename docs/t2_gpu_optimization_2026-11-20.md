# xchplot2 — T2 GPU高速化 実装・検証記録

- **実施日:** 2026-11-20
- **リポジトリ:** https://github.com/akamaya/xchplot2
- **対象ファイル:** `src/gpu/T2OffsetsSycl.cpp`
- **目的:** T1高速化済みのSYCL版xchplot2に、T2 Local CompactionとTezcan AES、共有メモリ・バンク分散を追加し、T2 matchを短縮する。
- **環境:** Windows / AdaptiveCpp (SYCL) / NVIDIA RTX 4070 SUPER 12 GB（計測は`gpu0`）
- **代表条件:** `k=28`, `strength=6`, 1 GPU, warmup 1 + measured 3 plots, T1 Compact ON, T2 Compact ON, T2/T3 Two-phase OFF

## 1. 実装の概要

### 1.1 T2 Local Compaction（`POS2GPU_T2_COMPACT=1`）

従来のT2単一カーネルでは、スレッドごとに候補数が異なるため、Pairing AESの実行回数がレーン間で不均一になる。T1で実証したLocal CompactionをT2にも適用した。

- 対象は通常の全バケット経路と、低VRAM向けのsection-pair split経路の2種類。
- ワークグループは256スレッド。各スレッドが一致候補数を計数し、`sycl::exclusive_scan_over_group`で書き込み先を算出。
- 候補の左右`uint64_t`メタデータを共有メモリ上の配列へ集約し、連続するレーンにPairing AESを分担させる。
- 候補ペアのローカル容量は512件（左右の2×`uint64_t`で約8 KiB）。512件を超える場合は、取りこぼしを避けるため従来型の候補ループにフォールバックする。
- 出力の`meta` / `mi` / `xbits`と、atomic counterによる出力確保の基本動作を維持。元の経路は削除せず、環境変数で実験経路を選択する。

### 1.2 T2 Tezcan AES（`POS2GPU_T2_TEZCAN=1`）

- AESのT0テーブルを16重に複製し、1,024エントリの従来4テーブル配置ではなく、`16 × 256`エントリ（16 KiB）の共有メモリ領域を利用する。
- T1で導入したTezcan方式と同様、T0の値を利用して残りのテーブル相当の演算をビット回転で求める。
- Compactionとの併用に加え、Compactionを使わずTezcan AESのみを有効にする実験経路も用意した。
- `POS2GPU_T2_COMPACT=1`の場合はCompaction経路を優先し、その中のAES方式を`POS2GPU_T2_TEZCAN`で切り替える。

### 1.3 共有メモリ・バンク分散修正

初期のT2 Compaction + Tezcanでは、16重複製テーブルの参照先が常に`bank=0`で固定されていた。32バンク構成・32 bitエントリを前提とすると、`(16 * index) % 32`となり、参照先が実質2バンクへ偏るため、バンク競合を起こしやすい。

これに対してCompaction内のTezcan呼び出し4か所を次のように修正した。

```cpp
static_cast<uint32_t>(local_id) & 15u
```

- `t2c_tezcan_target`：2か所（当日のローカルソース行529、1010）
- `t2c_tezcan_pairing`：2か所（当日のローカルソース行554、1041）

参照インデックスへレーンごとのバンク番号を加えることで、16複製を活用する。**この修正後に大幅な性能回復を観測したが、ハードウェア性能カウンタで競合数を直接測定したわけではないため、競合を唯一の原因と断定しない。**

### 1.4 環境変数と既存経路

| 環境変数 | 実測時の値 | 用途 |
|---|---|---|
| `POS2GPU_T1_COMPACT` | `1` | T1のLocal Compaction |
| `POS2GPU_T2_COMPACT` | `1` | T2のLocal Compaction（未指定時はOFF） |
| `POS2GPU_T2_TEZCAN` | `0` / `1` | T2のAES方式のA/B比較（未指定時はOFF） |
| `POS2GPU_T2_TWOPHASE` | `0` | 既存のT2二段階処理を無効化 |
| `POS2GPU_T3_TWOPHASE` | `0` | 既存のT3二段階処理を無効化 |
| `POS2GPU_PHASE_TIMING` | `1` | フェーズ別計測を出力 |

元のT2単一カーネル・二段階経路を保持。今回の最速結果は、**明示的にT2 CompactionとTezcanをONにし、T2/T3二段階処理をOFFにした条件**での結果であり、既定設定の性能を意味しない。

## 2. 性能計測結果

### 2.1 同一ビルドでのTezcan ON/OFF比較

同じ`k28 / strength6 / gpu0 / warmup 1 / measured 3`で比較。T2 Compactionは常にON。

| 指標 | T2 Tezcan OFF | T2 Tezcan ON | 変化 |
|---|---:|---:|---:|
| T1 match | 約7.79 s | 約7.79 s | ほぼ不変 |
| **T2 match** | **約12.71 s** | **約9.79 s** | **約2.92 s短縮（約23.0%）** |
| T3 match + Feistel | 約17.68 s | 約17.66 s | ほぼ不変 |
| **steady-state** | **39.72 s/plot** | **36.75 s/plot** | **2.97 s短縮（約7.5%）** |
| 月間生成量（30日） | 59.5 TiB | 64.3 TiB | 約8.1%増 |
| VRAMピーク | 10,474 MiB | 10,474 MiB | 変化なし |

- OFFのT2 match：測定3個のフェーズ値は12.7064 / 12.7097 / 12.7066秒。
- ONのT2 match：測定3個のフェーズ値は9.7641 / 9.7922 / 9.8007秒。
- ONのsteady-stateでの測定間隔は36.40〜36.98秒、σ=0.31秒。
- 実測時のGPUの利用可能VRAMは11,069 MiB、最小空きは595 MiB。余裕は大きくない。

### 2.2 T1高速化までの既存基準との参考比較

| 段階 | T2 match | steady-state |
|---|---:|---:|
| T1高速化済み、T2/T3二段階OFF（過去計測） | 17.531 s | 44.57 s/plot |
| T2 Compaction ON / Tezcan OFF（今回） | 約12.71 s | 39.72 s/plot |
| T2 Compaction ON / Tezcan ON + バンク分散（今回） | **約9.79 s** | **36.75 s/plot** |

過去基準と比較するとPlot全体で約17.5%、T2 matchで約44%短縮。ただし過去基準は同一セッションの厳密なA/B比較ではないので、今回のON/OFF比較を主な評価根拠とする。

修正前のT2 Compaction + TezcanはT2 matchが約60.6秒まで悪化した。バンク分散修正後は約9.79秒へ改善した。これはバンク競合仮説を強く支持するが、修正前後の完全な条件一致や性能カウンターの直接確認までは記録していない。

## 3. 正当性検証

### 3.1 CPU/GPU一致テスト：30/30 PASS

実行例：

```powershell
$env:POS2GPU_T2_COMPACT="1"; $env:POS2GPU_T2_TEZCAN="1"; $env:POS2GPU_T2_TWOPHASE="0"; .\build\windows-sycl\tools\parity\t2_parity.exe
```

- k18、Strength 2：20ケース（固定Plot IDおよび複数Seed）。
- k18、Strength 3〜7：各2ケース（計10ケース）。
- 全30ケースでCPUとGPUのT2件数および`mi` / `xbits` / `meta`が一致。
- 最終出力：`==> ALL OK`。

### 3.2 シャード一致テスト：6/6 PASS

```powershell
.\build\windows-sycl\tools\parity\sycl_t2_phase_sharded_parity.exe
```

- k18 / k20 / k22 × Seed 7 / 31 = 6ケース。
- 単一GPU基準とシャード分割経路の`count` / `mi` / `meta` / `xbits`が全件一致。
- 全6ケースで`PASS`。

### 3.3 完成PlotのProof検証：126/126成功

`bench --keep`によって生成した`bench-*.gplot`（k28 / Strength 6 / T2 Tezcan ON）に対して検証を実施した。

```powershell
.\build\windows-sycl\tools\xchplot2\xchplot2.exe verify <GPlotファイルのパス> --trials 100 --full
```

検証ログ：

```text
[verify] 100 trials, 50 with >=1 quality chain, 126 quality chains total
[verify] 126 full proofs validated
[verify] OK (full proof validation)
```

`bench`で生成するPlotは**合成・ファーミング不可**のテスト用。既定では出力ファイルを削除するため、保存・検証には`--keep`が必要。拡張子は`.gplot`。

## 4. 再現用のベンチマーク

Windows PowerShell（コマンドは1行ずつ実行）：

```powershell
cd C:\dev\xchplot2
$env:POS2GPU_T1_COMPACT="1"; $env:POS2GPU_T2_COMPACT="1"; $env:POS2GPU_T2_TWOPHASE="0"; $env:POS2GPU_T3_TWOPHASE="0"; $env:POS2GPU_PHASE_TIMING="1"; $env:POS2GPU_T2_TEZCAN="1"
.\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 3 --warmup 1 --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test
```

Tezcan OFFとの比較時は`$env:POS2GPU_T2_TEZCAN="0"`のみ変更し、それ以外を固定する。

## 5. 適用範囲・残課題

1. 実験経路は明示的に環境変数を指定して初めて使用される。既定で最速経路になるわけではない。
2. CPU/GPU一致テストはk18、シャード一致はk18/20/22。完成PlotのFull Proof検証はk28/Strength 6の1ファイル・100チャレンジで実施した。
3. T2二段階経路の最適化や、異なるGPU・Strength・VRAM tierでの性能・正当性検証は別作業。
4. Tezcanを含むCompactionの性能は良好だが、共有メモリのバンク競合数やoccupancyを性能カウンターで直接計測したわけではない。
5. 今回の修正対象はT2。T3の高速化は次の段階で実施する。

## 6. コミット対象

- `src/gpu/T2OffsetsSycl.cpp`：T2 Local Compaction、Tezcan AES経路、レーン別バンク分散。
- `docs/t2_gpu_optimization_2026-11-20.md`：本記録。

**推奨コミットメッセージ:**

```text
perf(sycl): optimize T2 with local compaction and banked Tezcan AES
```

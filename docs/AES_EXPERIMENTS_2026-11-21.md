# AES高速化の探索記録 — 採用しなかった方式も含む（2026-11-21）

## 目的・結論

xchplot2 の Windows 11 / AdaptiveCpp SYCL (CUDA backend) における T1/T2/T3 のAES計算の高速化候補を、**成功・不成功を問わず**保存する。実験対象の大半は `tools/parity/sycl_aes_bench.cpp` の**AES単体マイクロベンチ**であり、`src/gpu/T*OffsetsSycl.cpp` の本番Plot時間やアルゴリズム変更と同一視しない。

- GPU：**NVIDIA GeForce RTX 4070 SUPER（12 GB）**。原則 `262144` AES states、256 AES round-key *pairs* = **512 AES rounds**、または16 pairs = **32 AES rounds**。BitSlice初期試験のみ `65536` states。
- 既存高速実装：**Rotate16**。AES T0を16複製（16 KiB shared memory）、T1/T2/T3相当の値を32bit回転で再構成。元の4テーブルT-table版を出力一致の基準とした。
- **採用：** 既存の16複製T0、V4INIT（128bit初期化）、今回の **Shared16 direct-address（T2/T3のみ）**。その他の候補は不採用または改善未確認。V4INITの詳細は [`AES_V4INIT_OPTIMIZATION_2026-11-21.md`](AES_V4INIT_OPTIMIZATION_2026-11-21.md) を参照。
- 以下の数値は過去の対話・ベンチログ・実験READMEから回収した記録。**元のログが現在のリポジトリで公開されていない値は「記録値」として扱い、厳密な再現・統計保証はしない**。条件が違う表の数値を直接比較しない。

### 結果一覧

| 系統 | 比較条件・代表値 | 判定 | 本番Plot A/B |
|---|---|---|---|
| BitSlice（32レーン協調） | 65,536 states、512 rounds：Rotate16 **0.284 ms** / BitSlice **0.575 ms** | **不採用**。約2.02倍の時間 | 記録なし |
| ILP2 / ILP4 / ILP8 | 262,144 states、512 rounds：Rotate16 **1.135 ms** / ILP2 **1.180** / ILP4 **1.186** / ILP8 **1.385 ms** | **不採用** | 記録なし |
| MIO対応（512スレッド・GlobalT0） | 512 rounds：Rotate16 約**1.031 ms** / R16_512 **1.082** / R32_512 **1.089** / GlobalT0 **1.610 ms** | **不採用** | 記録なし |
| 384スレッド・ループ展開 | U16_8 **1.028 ms** 対 Rotate16 **1.037 ms**（512 rounds、代表値） | **効果未確認**。約0.9%はばらつきと区別不可 | 記録なし |
| Odd Bank / 複製数12～20 | 512 rounds：**1.661～1.796 ms**（7候補） | **全候補不採用** | 記録なし |
| 4つのT-table × 複製 | 512 rounds：最速T4_R8 **1.495 ms** 対 Rotate16 **1.029 ms** | **全候補不採用** | 記録なし |
| Packed S-box | 512 rounds：SP16 **2.042 ms** / SP32 **1.857 ms** 対 Rotate16 **1.031 ms** | **両候補不採用** | 記録なし |
| Direct Address（Shared16） | 512 rounds：約1.02～1.03ms。長時間A/Bで平均約1.26%短縮 | **T2/T3採用**、T1は従来版 | **phase合計1.03%短縮** |
| Shared16各種命令順序変更 | Parity PASS、追加改善の再現性なし | 不採用 | 記録なし |
| Shared/Global固定50:50 | 512 rounds：1.329 / 1.554ms | 不採用 | 記録なし |
| index bit分割Hybrid | 512 rounds：2.091～2.480ms | 不採用 | 記録なし |
| PairTable（256KiB） | 512 rounds：17.72～19.54ms | 不採用 | 記録なし |

> 時間の表示は原則として実験プログラムが報告した経過時間。**512 rounds** は `sycl_aes_bench.exe ... 256`、**32 rounds** は `... 16` を意味する。ホストからの起動・同期等を含む計測値なので、純GPU命令のレイテンシを示すものではない。

---

## 1. BitSlice（採用せず）

**狙い：** T0のshared-memoryランダムロードを排除し、Warp（32レーン）間でAESビット演算を分担する。局所集約中のT1 pairing経路には実験用 `POS2GPU_T1_BS_PAIR=1` も用意したが、正式採用していない。

### RTX 4070 SUPERでの実測（**65,536 states**）

| AES rounds | Rotate16 | BitSlice | BitSlice / Rotate16（時間比） | 一致判定 |
|---|---:|---:|---:|---|
| 32 | 0.039 ms | 0.091 ms | **2.33倍遅い** | 0 mismatches |
| 512 | 0.284 ms | 0.575 ms | **2.02倍遅い** | 0 mismatches |

**別条件（262,144 states、512 rounds、Nsight実行時）：** Rotate16 **1.135 ms**、BitSlice **2.132 ms**（約1.88倍遅い）。GPUのT1パリティ試験では **261,538 / 261,538 pairing一致**と報告された。一方、**BitSlice ONのk28実Plot速度A/Bが完了した記録はない**。

**判定：** 共有メモリ参照を減らす代わりに、論理演算・レーン間処理が増え、この条件ではRotate16より低速。T1に本採用せず。原因の内訳は未計測。

再現関連：`add_aes_bitslice_bench.py`、`aes_bitslice_sycl_benchmark.patch`、`enable_t1_bitslice_pair.py`、`AES_experiment_README_20261121.md`（実験時の作業ファイル）。

## 2. ILP2 / ILP4 / ILP8（採用せず）

**狙い：** 1スレッドに独立したAES状態を2/4/8本持たせ、連続するAESの依存チェーンを重ねて命令並列性を上げる。AES本体は既存Rotate16を維持。

### RTX 4070 SUPERでの実測（262,144 states、512 rounds）

| 方式 | 同一ログ内の代表測定 | 別ログでの観測範囲 | 一致判定 |
|---|---:|---:|---|
| Rotate16 | **1.135 ms** | 1.125～1.135 ms | 0 mismatches |
| ILP2 | 1.180 ms | 1.180～1.210 ms | 0 mismatches |
| ILP4 | 1.186 ms | 1.183～1.195 ms | 0 mismatches |
| ILP8 | 1.385 ms | 1.385～1.392 ms | 0 mismatches |

実測により**独立チェーンを増やすほど必ず速くなるわけではない**と確認。ILP2/4はRotate16を上回れず、ILP8は明確に遅い。レジスタ使用量や命令数・ワーク分配変化が考えられるが、個々の遅延要因はこの比較だけでは確定しない。

再現関連：`add_aes_ilp_bench.py`、`AES_ILP_README_20261121.md`。上述のベンチ値は過去に保存された `profile_aes_ncu.ps1` 実行出力から確認できる。実Plotへの移植は不採用。

## 3. MIO / Occupancy / Shared Memory 512スレッド・GlobalT0（採用せず）

**狙い：** Nsight Computeで見えたMIOパイプライン圧迫を軽減する。`Rotate16` と比較して、ブロック512スレッドの `R16_512` / `R32_512`、およびshared複製を使わず1KiBの元T0をglobalキャッシュ経由で読む `GlobalT0` を試した。

### Nsight Compute観測（**262,144 states、512 rounds**）

| 項目 | Rotate16 | Rotate32 | ILP2 |
|---|---:|---:|---:|
| Registers / thread | 39 | 39 | 40 |
| Dynamic shared memory / block | 16.38 KiB | 32.77 KiB | 16.38 KiB |
| Achieved occupancy | **74.74%** | **47.34%** | **68.39%** |
| Memory pipes busy | 95.56% | 95.41% | 90.42% |
| MIO queue throttle（issued instruction間のstall割合） | **47.0%** | 39.8% | 46.6% |

- `Rotate16` のShared LoadはNsightの命令集計で**約2.0005 Wavefront / LDS命令**。Bank Conflictも約1.0004 / LDS。共有メモリ側の処理仕事量が増える。ただしAES全体が2倍遅いという意味ではない。
- `Rotate32` は32KiB超のshared確保でoccupancy低下。Bank Conflict削減だけでは総合性能が伸びず、AES単体のRotate16との差もほぼなかった。
- DRAM帯域が最大まで埋まった証拠はない。MIO stallはsharedアクセス**だけ**に起因するとは限らない（特殊命令・分岐等を含む）。

### AES単体の追加候補（262,144 states、512 rounds、実験記録値）

| 方式 | 記録値 | 比較結果 | 一致判定 |
|---|---:|---|---|
| Rotate16（この比較系列の基準） | 約1.031 ms | 基準 | 0 mismatches |
| R16_512 | 1.082 ms | 遅い | 0 mismatches |
| R32_512 | 1.089 ms | 遅い | 0 mismatches |
| GlobalT0 | **1.610 ms** | 大幅に遅い | 0 mismatches |

**判定：** 512スレッド化や単純なGlobalT0参照は不採用。GlobalT0はテーブルが小さくても、L1ヒットとshared-memoryロードが同じコストになるわけではない。**新しいキャッシュ実験を行う場合は、既にGlobalT0 1.610msで負けた事実と何を変えるのかを先に記録する。**

資料：`AES_MIO_README_20261121.md`、`aes_rotate16_ncu_analysis_20261121.md`、Nsight Computeの実測CSV/レポート。

## 4. 384スレッド、ループ展開U16_1/4/8（改善確証なし）

**狙い：** 256→384スレッド/ブロック（`R16_384`,`R32_384`）、およびペアAESループの展開数（`U16_1`,`U16_4`,`U16_8`）を変え、分岐/ループ命令と占有率のバランスを調べる。

- 512 roundsの記録：`U16_8` **1.028 ms**、比較時の`Rotate16` **1.037 ms**。時間短縮約**0.9%**。
- 実験総括では、`R16_384`、`R32_384`、`U16_1`、`U16_4` に**再現性のある改善は確認されなかった**。
- **未回収：** 上記4候補の正確な各ms、32 rounds比較の各ms、繰り返し測定のばらつき。元ログが必要。**数値を作らない。**
- 出力一致は一連のAES実験で確認されたとの報告があるが、各候補別の生ログはこの記録では未回収。全候補の0 mismatchesを独立に検査済みと断定しない。

**判定：** `U16_8` の0.9%短縮は小さく、条件変更の効果・計測変動を切り分けられない。AES本体への採用なし。

再現関連：`add_aes_unroll_384_bench.py`、`AES_UNROLL_384_README_20261121.md`。

## 5. Odd Bank：B12/B14/B15/B17/B18/B19/B20（採用せず）

**狙い：** 16複製と32複製の中間や奇数strideで共有メモリ使用量・バンク配置・アドレス計算を変える。ワークグループは256スレッド。

### 実測（262,144 states、512 rounds / 32 rounds）

| 候補（T0複製数） | 512 rounds | 32 rounds | 出力一致 |
|---|---:|---:|---|
| B12 | 1.661 ms | 0.124 ms | 0 mismatches |
| B14 | 1.753 ms | 0.129 ms | 0 mismatches |
| B15 | 1.786 ms | 0.132 ms | 0 mismatches |
| B17 | 1.794 ms | 0.133 ms | 0 mismatches |
| B18 | 1.759 ms | 0.131 ms | 0 mismatches |
| B19 | 1.796 ms | 0.134 ms | 0 mismatches |
| B20 | 1.677 ms | 0.127 ms | 0 mismatches |

この比較時のRotate16はおおむね **1.03 ms（512 rounds）/ 0.087～0.09 ms（32 rounds）**。**すべてRotate16より遅かった**。複製数の選択によってアドレス計算とメモリバンク分散が悪化した可能性があるが、候補別のbank-conflictカウンタは記録なし。奇数strideが必ず衝突回避になるわけではない。

再現関連：`add_aes_odd_bank_bench.py`、`AES_ODD_BANK_README_20261121.md`。

## 6. 4テーブル複製方式：T4_R1/R2/R4/R8（採用せず）

**狙い：** T0～T3の4テーブルを使って、Rotate16方式の回転演算（1 AES roundで最大12回程度）を省く。テーブルは4KiB×複製数。T4_R4は合計16KiBでRotate16と同じ容量。

### 実測（262,144 states、512 rounds / 32 rounds）

| 候補 | Shared memory / block | 512 rounds | 32 rounds | 出力一致 |
|---|---:|---:|---:|---|
| T4_R1 | 4 KiB | 1.613 ms | 0.119 ms | 0 mismatches |
| T4_R2 | 8 KiB | 1.671 ms | 0.123 ms | 0 mismatches |
| T4_R4 | 16 KiB | 1.644 ms | 0.127 ms | 0 mismatches |
| T4_R8 | 32 KiB | **1.495 ms** | 0.126 ms | 0 mismatches |
| Rotate16（同条件の記録値） | 16 KiB | **約1.029 ms** | **約0.087 ms** | 0 mismatches |

**判定：** 演算回転を減らしても、sharedメモリ参照・レジスタ/命令・レーン間の衝突等を含めた実行時間は悪化。何が主要原因かは候補別NCUがなく断定不可。全候補不採用。

再現関連：`add_aes_four_tables_bench.py`、`AES_FOUR_TABLES_README_20261121.md`。

## 7. Packed S-box：SP16/SP32（採用せず）

**狙い：** 4つのS-box値を1つの32bitに詰め、MixColumnsを計算で再構成して、テーブル複製容量を減らす。

- `SP16`：64×16×4B = **4KiB** shared memory。
- `SP32`：64×32×4B = **8KiB** shared memory。32レーン分散でbank conflict低下を狙う。

### 実測（262,144 states、512 rounds / 32 rounds）

| 候補 | 512 rounds | 32 rounds | 出力一致 |
|---|---:|---:|---|
| SP16 | **2.042 ms** | **0.147 ms** | 0 mismatches |
| SP32 | **1.857 ms** | **約0.137～0.140 ms** | 0 mismatches |
| Rotate16（同条件の記録値） | **約1.031 ms** | **約0.088 ms** | 0 mismatches |

Pythonの式検査はランダムなAES状態とキー **100,000件**で一致したとの記録あり。Windows GPUの実行結果でも両方式の0 mismatchesが報告された。

**判定：** SP32はSP16より速いが、Rotate16に対して512 roundsで約**80.1%時間増**。MixColumnsの再構成に必要な演算増加が効いた可能性がある（内訳のSASS/NCU測定は未回収）。両方式とも不採用。

再現関連：`add_aes_packed_sbox_bench.py`、`AES_PACKEDSBOX_README_20261121.md`。

---

## 8. 既に採用した方式との境界

- **採用：** AES T0の16複製Rotate16（既存）、`V4INIT` 128bit共有メモリ初期化（T1～T3、12箇所）、および**Direct Address（T2/T3 Tezcan経路のみ）**。これらはAES計算中の約2 Wavefront/Shared Loadという問題自体は変更しない。
- **別の採用済み最適化：** Local Compaction、T2/T3 Fine Bucket 16+4直接参照、候補容量 T1=512/T2=480/T3=330。AES単体ベンチ値と、実Plotの改善率を混同しない。
- V4INIT実Plotの同設定A/B：`phase-timing` OFF平均 **26,916.0 ms** / ON **26,450.4 ms**（**465.6 ms、1.73%短縮**）。k28 Strength6の合成Plotは `verify --full --trials 100` で **114/114 full proofs validated**。詳細は上記V4INIT資料。
- **未採用技術を再試験する条件：** 基準・GPU・round数・state数・ビルドSHA・温度/クロック等を揃え、変更した仮説を明示すること。以前の「GlobalT0は低速」「Rotate32はoccupancy低下」を理由なく繰り返さない。

## 9. どう測定し、何が分からないか

再現用の共通コマンド例（実験コードがローカルに残っている場合のみ）。実験拡張を複数適用していないバイナリでは候補行は表示されない。

```powershell
cmake --build .\build\windows-sycl --config Release --target sycl_aes_bench -j 16
.\build\windows-sycl\tools\parity\sycl_aes_bench.exe 262144 256
.\build\windows-sycl\tools\parity\sycl_aes_bench.exe 262144 16
```

- **時刻差の扱い：** 初回JITをウォームアップで除外。変動が2%未満なら複数回の交互A/Bや分散確認なしに「勝った」としない。
- **暗号の正当性：** `OK (0 mismatches)` はAES単体出力が既存の基準に一致すること。**Plot全体**の検証は同一入力CPU/GPUの出力比較と、保持した`.gplot`に対する`verify --full --trials 100`を分ける。
- **推定と計測の区別：** SIMD/レジスタ、shared銀行競合、occupancy低下の説明は、NCUカウンタを取得していない候補では仮説。試験値をベンチの速度改善や因果と断言しない。
- **元ログの所在：** 実験時の `AES_*_README_20261121.md` と各 `add_aes_*_bench.py` はローカル作業ファイル。各候補の数値が未回収のものは上記に明示し、未検証として将来補完する。

## 10. 今後の運用ルール

**採用/不採用に関係なく、各実験が終わった時点で必ず `docs/` に追記してからコミットする。** 新規実験は [`EXPERIMENT_RECORD_TEMPLATE.md`](EXPERIMENT_RECORD_TEMPLATE.md) を複製し、仮説、比較対象、機材・設定、ビルドSHA、各回の数値・ウォームアップ、パリティ、Nsight測定、採否と理由、未測定項目、次回の再検討条件を記録する。失敗実験こそ重複を避けるために残す。
---

## 11. Shared16 Direct Address・命令順序変更（T2/T3のみ採用）

**仮説：** 1 AES roundあたり16回のLDSと16KiBのT0複製を維持したまま、状態wordからbyte値を取り出して添字をスケーリングする代わりに、shared-memoryの**バイトオフセット**を直接計算する。レーン/複製を`replica=0..15`とすると：

```cpp
// 元は t[((word >> (8 * byte)) & 255u) * 16 + replica]
byte0_offset = ((word <<  6) & 0x3fc0u) | (replica << 2);
byte1_offset = ((word >>  2) & 0x3fc0u) | (replica << 2);
byte2_offset = ((word >> 10) & 0x3fc0u) | (replica << 2);
byte3_offset = ((word >> 18) & 0x3fc0u) | (replica << 2);
```

**結果：** 300 samples × 6組の交互マイクロベンチでは、すべての組で改善し平均**約1.26%**短縮（当時の記録値）。ただし少数サンプルでは差がばらつく。262,144 states / 512 AES roundsの通常測定例は`shared16=1.0345 ms`、`shared16-addr=1.0318 ms`、別の回では`1.0341 ms`→`1.0242 ms`。各回の基準値が異なるので跨って単純比較しない。

NCUで4 AES roundsの静的SASS命令数**340→276**、動的命令数**357,269,504→290,160,640**（約18.8%減）を観測。一方、Shared LDSの動的命令数はどちらも**67,108,864**、wavefrontはどちらも約**134.25M**。L1/TEXはほぼ飽和したまま。**SASS命令数が18.8%減っても実行時間は18.8%短縮しない**。

`shared16-early`、`shared16-word`、`shared16-prmt`、`shared16-bfe`、`shared16-addr-lop3`、`shared16-addr-bfi`、`shared16-addr-xor-early`、`shared16-addr-xor-tail`などはParity PASSだが、Direct Addressに対する再現性ある追加改善は未確認。個別の揃った測定値がないので作らない。

**判定：** Direct Addressのみ本番Plot A/Bへ進め、ほかの派生は不採用。

## 12. Shared/Globalの50:50 Hybrid（不採用）

`hybrid50-16k`は元16KiBのsharedテーブルを保持し16参照のうち8つをglobalへ、`hybrid50-8k`はsharedを8KiBにして同様に8つをglobalへ割り当てる固定分割。どちらもParity PASS（mismatches=0）。

| モード | 32 AES rounds | 512 AES rounds |
|---|---:|---:|
| `shared16`（同一測定回） | **0.0871 ms** | **1.0249 ms** |
| `hybrid50-16k` | 0.1040 ms | 1.3289 ms |
| `hybrid50-8k` | 0.1245 ms | 1.5542 ms |

**判定：** 全候補不採用。Global側のランダムアクセスが負担となった可能性があるが、Hybrid専用のNCUカウンタを取得していないので断定しない。

## 13. T0 indexのbitでShared/Globalを分離（不採用）

256エントリのT0をMSB（0..127 / 128..255）またはLSB（偶奇）で半分に分け、128エントリ×16複製=8KiBをShared、残りをGlobalへ。`512`接尾辞は512-thread/block版。すべてParity PASS（mismatches=0）。

| モード | 32 AES rounds | 512 AES rounds |
|---|---:|---:|
| `shared16`（同一測定回） | **0.0844 ms** | **1.0345 ms** |
| `hybridbit-msb` | 0.1593 ms | 2.0906 ms |
| `hybridbit-lsb` | 0.1672 ms | 2.3661 ms |
| `hybridbit-msb512` | 0.1643 ms | 2.1884 ms |
| `hybridbit-lsb512` | 0.1833 ms | 2.4801 ms |

**判定：** 全候補不採用。Warp内でShared/Globalの参照先が混在することによる命令発行の増加などが候補だが、NCU未測定につき原因は仮説。

## 14. 2参照を結合した256KiB PairTable（不採用）

`PairTable[(a << 8)|b] = T0[a] ^ rotl8(T0[b])`を65,536エントリ×4BのGlobalテーブルに事前計算して、16 LDS/AES roundを8 LDGに変更。256KiB自体はL2容量に入るが、32レーンにランダムな16bit indexを与えるためセクタアクセスは分散する。生成時間はベンチ計測対象外。全モードParity PASS。

| モード | 32 AES rounds | 512 AES rounds |
|---|---:|---:|
| `shared16-addr`（同一測定回） | **0.0841 ms** | **1.0242 ms** |
| `pair-global`（標準キャッシュ） | 1.2666 ms | **19.5448 ms** |
| `pair-global-ca`（L1+L2） | 1.2661 ms | **17.7223 ms** |
| `pair-global-cg`（L2） | 1.2663 ms | **17.7616 ms** |

**判定：** 約17.3～19.1倍遅いため不採用。`ca`/`cg`の違いは小さい。L1/L2ヒット率やセクタ数・帯域のNCU測定はないため、具体的な律速は未確定。**キャッシュに収まる=速いではない**ことを示す実測。

## 15. 本番Plot・Direct Address ABBA比較（T2/T3採用）

- **環境：** Windows 11 / RTX 4070 SUPER / AdaptiveCpp SYCL CUDA backend、2026-11-21 JST。
- **条件：** k28、Strength6、GPU0、NAS出力、`XCHPLOT2_BACKEND=sycl`、`POS2GPU_FINE_DIRECT=1`、Fine bits T1/T2/T3=8/16/16、Compact全ON、T2/T3 Tezcan全ON、T2/T3 TwoPhase全OFF、`POS2GPU_PHASE_TIMING=1`。
- **比較：** 2種類のコンパイル時固定バイナリ。**実験時はT1/T2/T3すべてを切り替え**、AESラウンド内の実行時分岐なし。Baseline→Optimized→Optimized→BaselineのABBA順で、各回warmup1 Plotを除外し測定2 Plot。各方式4 Plotを集計。
- **実験ログ：** `bench_local/aes_prod_addr_ab_20261121-221513/`（手元に保管、ログ本体はGit未収録）。

| フェーズ | Baseline平均 | Direct Address平均 | 改善率（短縮率） |
|---|---:|---:|---:|
| T1 match | 7,827.4 ms | 7,839.5 ms | **-0.15%** |
| T2 match | 8,510.5 ms | **8,392.3 ms** | **1.39%** |
| T3 match + Feistel | 8,812.8 ms | **8,672.5 ms** | **1.59%** |
| **phase合計** | **26,488.5 ms** | **26,216.7 ms** | **1.03%**（約272ms） |

各回の測定2 Plot平均（phase合計）：Baseline 1回目 **26,477.1ms** / Optimized 1回目 **26,212.0ms** / Optimized 2回目 **26,221.4ms** / Baseline 2回目 **26,500.0ms**。Baselineへ戻すと再び時間が増加する。NAS書き込みやGPU状態を含む`steady-state`値は参考に留め、上表の`phase-timing`を主な判断材料にした。

**正式採用：** T2/T3の**Tezcan AESのみ**Direct Addressに置換する。改善がなかった**T1は従来のアドレス計算を維持**。Shared16（16KiB）、LDS 16/round、V4INIT、4-tableの非Tezcan経路、Fine Bucket、Native CUDAは変更しない。実験時のcompile-time ON/OFF分岐は本番に残さない。

**正当性：** 実験時のCPUでの10万ランダム状態による演算一致を確認。ユーザーはDirect Address版で生成したPlotについて`verify --full --trials 100`のPASSを報告した（検証ログの証明個数は未提供なので記載しない）。**最終コードはT1を元の実装にしたT2/T3限定構成**なので、この最終構成の実Plot時間・proof検証はまだ未実施。上の**1.03%は実験版（T1/T2/T3すべてON）の実測**であり、最終T2/T3限定版の測定値ではない。

## 16. 結論・再試験ルール

- **採用：** Shared16（16KiB）、V4INIT（128bit初期化）、Direct Address（T2/T3 Tezcan）。
- **不採用：** BitSlice、ILP2/4/8、Rotate32・GlobalT0・512threads、Odd Bank、4-table、Packed S-box、Hybrid50、HybridBit、PairTable、および追加改善を確認できなかったDirect Address派生。
- **未確定：** 384-thread/Unroll8の小改善、T2/T3限定最終実装でのPlot再測定・proof再検証。
- 失敗候補を再試験するときは、前回から何を変えるのか、Parity、同条件A/B、必要に応じてNCUの測定項目を明確にする。AES単体ベンチだけで実Plot採否を決めない。

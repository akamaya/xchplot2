# Fine Bucket 最適化記録 — T2/T3 16+4bit 圧縮直接参照

- 対象: `akamaya/xchplot2`、Windows 11 / AdaptiveCpp SYCL (CUDA backend)
- 作業完了日: 2026-11-21 (JST)
- 計測環境: RTX 4070 SUPER 12 GiB (`gpu0`)、`k=28`、Strength 6、GPUワーカー1、出力先NAS (SMB)
- 状態: **10 plot の速度ベンチマーク完走・暫定最速として採用。新方式固有のCPU/GPU parity と full-proof 検証は未完了。**
- 適用範囲: **SYCL版 T2/T3 の all-buckets `COMPACT=1` 経路のみ**。T1はFine Bits=8のまま。CUDAネイティブ・two-phase・他の経路は従来の検索のまま。

## 1. 背景と目的

T2/T3のマッチングでは、左側候補 (`meta_l`) と右側のmatch keyからAESターゲットを生成し、ソート済みの右側 `match_info` から **下位20bitが完全一致する全候補** を探す。`k=28, strength=6` では Section が4通り、Match Key が64通りなので、右側Bucketは `4 × 64 = 256` 個ある。

従来はBucketごとに上位 `N` bit のFine索引を用意し、残り `20-N` bitに対して二分探索 (`fine_bucket_lower_bound`) を行っていた。Fine Bitsを16へ拡張すると1検索あたりの二分探索は約4ステップまで減るが、T2/T3の検索呼び出し回数が非常に多いため、さらに完全一致候補範囲を直接引くことで検索負荷を減らす。

**注意:** `[fine-timing] match_fused` は検索以外にもAESターゲット生成、候補集約、ペアリングAES、T3のFeistelなどを含む。後述の短縮量を「二分探索だけにかかった時間」とは解釈しない。

## 2. 実装した方式

16bit Fine索引を引き続き利用し、**下位4bitの開始位置を1byteで圧縮したテーブル**をその後ろに追加する。

```text
20bit target = [上位16bit Fine key][下位4bit suffix]

fine_idx = (r_bucket << 16) | (target >> 4)
base     = fine_offsets[fine_idx]            # u64、元のFine索引
end      = fine_offsets[fine_idx + 1]
rel      = suffix_start[fine_idx*16 + (target & 0xf)]

if rel != 0xff:
    first = base + rel
    last  = (suffix == 15) ? end : base + suffix_start[fine_idx*16 + suffix + 1]
    # [first,last) は20bit完全一致の候補のみ。二分探索不要。
else:
    first = lower_bound(sorted_mi, base, end, target)
    last  = end
    # 従来どおり target 完全一致を確認しながらループ
```

索引構築は `fine_offsets` 完了後にGPUで1回行う。各16bitグループについて、ソートされた行を1回前進走査しながら下位4bit `0..15` の16個のlower-boundを保存する。`sorted_mi` の重複値も、同じ開始・終了区間に含まれる。

### 高密度グループの安全なフォールバック

16bitグループの件数が **255件以上** の場合、相対位置を1byteに安全に格納できないので16個すべてを `0xFF` にする。そのグループへの検索は従来の `fine_bucket_lower_bound` に戻す。存在しないターゲット、同一値の重複、空グループ、および偏った分布でも一致候補を落とさない設計。

### メモリ量（T2かT3の1フェーズ分）

| 領域 | 数式 | 容量 |
|---|---|---:|
| Bucket offsets | `(256+1) × 8` | 2,056 bytes |
| 上位16bit Fine offsets | `(256×2^16+1) × 8` | 約128 MiB |
| 下位4bit suffix starts | `256×2^16×16×1` | 256 MiB |
| **計（実測サイズ）** | | **402,655,248 bytes ≈ 384 MiB** |

64bitのFineオフセットを維持しているため、以前検討した「32bit化で320MiB」は**まだ実装していない**。T2とT3は同時にこの領域を使わず、プールの `d_sort_scratch` を時間差で共有する。プール容量はソート一時領域とT1/T2/T3のmatch-prepare要求を比較し最大値で確保する。

## 3. 設定と対象実装

| 環境変数 | 値 | 動作 |
|---|---|---|
| `POS2GPU_FINE_DIRECT` | `1` | 16+4bit直接参照を有効化 |
| `POS2GPU_FINE_DIRECT` | `0` または未設定 | 従来のFine索引＋二分探索（追加suffixテーブルも作らない） |
| `POS2GPU_T1_FINE_BITS` | `8` | この検証ではT1を固定 |
| `POS2GPU_T2_FINE_BITS` | `16` | T2の上位16bit索引 |
| `POS2GPU_T3_FINE_BITS` | `16` | T3の上位16bit索引 |
| `POS2GPU_T2_COMPACT`, `POS2GPU_T3_COMPACT` | `1` | 新方式が適用されるSYCLのall-buckets経路を選ぶ |
| `POS2GPU_T2_TWOPHASE`, `POS2GPU_T3_TWOPHASE` | `0` | 計測時の設定 |
| `POS2GPU_FINE_TIMING`, `POS2GPU_PHASE_TIMING` | `1` | 詳細タイミング表示（性能に影響し得る診断モード） |
| `POS2GPU_POOL_DEBUG` | `1` | 共用一時領域のサイズを表示 |

**制約:** `POS2GPU_FINE_DIRECT=1` は T2/T3両方が `fine_bits=16` かつ `num_match_target_bits=20` の場合のみ対応し、それ以外は例外で終了する。この条件は今回の `k=28, strength=6` に対応する。デフォルトではOFF。

関連ソース（今回の一連の作業で変更／追加されたファイル）:

- `src/gpu/FineBucketConfig.hpp`: Fine Bits設定
- `src/gpu/FineBucketDirect.hpp`: 圧縮suffix索引の構築／検索／高密度フォールバック
- `src/gpu/T2Kernel.cpp`, `src/gpu/T3Kernel.cpp`: 索引領域サイズ、prepare時の構築
- `src/gpu/T2OffsetsSycl.cpp`, `src/gpu/T3OffsetsSycl.cpp`: COMPACTカーネル内の検索置換
- `src/host/GpuBufferPool.cpp`: ソート用scratchとT1/T2/T3索引サイズの最大値を確保
- `src/gpu/T1Kernel.cpp`, `src/gpu/T1OffsetsSycl.cpp`, `src/host/GpuPipeline.cpp`, `src/gpu/MatchDiag.hpp`: Fine Bitsの可変化／診断関連（同一開発系列）

## 4. 再現手順（PowerShell、各行そのまま実行）

既存のWindows SYCLビルドと、Fine Bits・プール容量修正・16+4実装がソースへ反映済みであることが前提。リポジトリルートで実行する。出力先は使用環境に合わせて置き換えること。

```powershell
cmake --build .\build\windows-sycl --config Release --target xchplot2 -j 12
$env:POS2GPU_T1_FINE_BITS='8'; $env:POS2GPU_T2_FINE_BITS='16'; $env:POS2GPU_T3_FINE_BITS='16'; $env:POS2GPU_T1_COMPACT='1'; $env:POS2GPU_T2_COMPACT='1'; $env:POS2GPU_T3_COMPACT='1'; $env:POS2GPU_T2_TWOPHASE='0'; $env:POS2GPU_T3_TWOPHASE='0'; $env:POS2GPU_FINE_DIRECT='1'; $env:POS2GPU_MATCH_DIAG='0'; $env:POS2GPU_FINE_TIMING='1'; $env:POS2GPU_PHASE_TIMING='1'; .\build\windows-sycl\tools\xchplot2\xchplot2.exe bench -k 28 -s 6 -n 10 --warmup 1 --devices gpu0 -o \\NAS-UGREEN\personal_folder\plots_test 2>&1 | Tee-Object fine_direct_16plus4_n10.log
```

従来16bitとのA/B比較では、**上のコマンドの `POS2GPU_FINE_DIRECT='1'` だけを `'0'` に変更**し、同じGPU、出力先、その他の環境変数で実行する。シェルに残るT2/T3 Tezcan等の設定は各比較で明示的に揃えること。`bench` の `.gplot` は合成ベンチマークで農業用の実Plotではない。

## 5. 実測結果（2026-11-20～21 JST）

**全条件** `k=28, strength=6, -n 10, --warmup 1, gpu0`。各行のmatch/phaseはログの**先頭ウォームアップ1件を除く10件**の平均。`phase-timing`合計にはT1/T2/T3以外のXs生成・sort・D2H転送も含む。`steady-state` はプログラムの出力値で、出力処理を含む実効間隔。タイミング診断はON。

| 構成（T1/T2/T3） | T2 `match_fused` | T3 `match_fused` | `phase-timing`合計 | Steady-state |
|---|---:|---:|---:|---:|
| 8/8/8（二分探索） | 9,821.1 ms | 10,040.4 ms | 28,907.9 ms | 28.74 s/plot |
| 8/14/14（二分探索） | 9,448.9 ms | 9,622.8 ms | 28,133.1 ms | 28.16 s/plot |
| 8/16/16（二分探索） | 9,273.1 ms | 9,455.6 ms | 27,809.7 ms | 27.95 s/plot |
| **8/16/16（16+4bit直接参照）** | **9,094.9 ms** | **9,248.4 ms** | **27,404.5 ms** | **27.27 s/plot** |

**改善**: 直接参照方式は従来16bitに対し `0.68 s/plot` 短縮（約2.43%高速化）。初期8bitに対し `1.47 s/plot` 短縮（約5.11%時間短縮）。各Plotの `direct_build` はT2平均 **6.785ms**、T3平均 **6.777ms**（ウォームアップ除外）。2フェーズ計約13.56msで、検索側の短縮より小さい。

**観測ログ:** `T2 bits=16 direct_build=...`、`T3 bits=16 direct_build=...`、`[pool] match-index bytes: T2=402655248 T3=402655248; shared sort_scratch=402655248`。約384MiBの索引を正常に確保し、10件ベンチが完走した。

**解釈上の注意**: 今回の比較は逐次実行であり、GPUクロック、温度、NASの書込混雑、ホスト側圧縮などにより値は変動する。スループットの差を更に厳密に見るには交互A/B/A実験を行う。`match_fused`とエンドツーエンドは別の指標であり、単純に短縮量を一致させない。

## 6. 検証・残課題

- [x] CPU側Python参照テスト: **58データセット×128検索**で、空、重複、偏り、255件境界などを確認（パッチ適用時 `--check`/`--apply`）。
- [x] Windows 11 / AdaptiveCpp SYCL実機でビルド、ウォームアップ＋10 Plotベンチが完走。
- [x] 16+4bit構成でソート用scratchを共有確保できた。
- [ ] **新方式ON/OFFで同一Plot ID・同一入力のGPU出力を比較するparityテスト**（出力順不同なのでソート後に比較）。
- [ ] **新方式で生成した実Plotに対して`verify --full`を実施**。ベンチ完走のみでは証明の正確性は保証されない。
- [ ] CUDA-native、tiny/minimal、two-phaseなど別経路への適用・検証。現時点ではそれらを「直接参照適用済み」と呼ばない。
- [ ] 実運用で診断フラグOFFの時のスループット再測定。余裕があれば32bit化で索引384→320MiBを検討。

## 7. ロールバック

実装コードを取り消さず、従来方式へ戻せる。

```powershell
$env:POS2GPU_FINE_DIRECT='0'
```

`POS2GPU_FINE_DIRECT=1` 時にだけ追加の256MiBを確保・構築する。VRAM条件が変わって別パイプラインへフォールバックした場合は、同じ方式・同じGPUの測定としては比較しないこと。

## 8. 開発履歴と出典

- Fine Bits可変化と`fine-timing`、診断（`match_diag`）を導入して検索の律速を調査。
- 8→14→16bitの実測後、16bitグループごとに下位4bitを1byteで持つ方式を導入。
- 初回の直接参照試験でJITコンパイル警告あり。ウォームアップを除いた10件で再評価。
- ベンチ元ログ: `fine_baseline_8_8_8_n3.log`（コマンド内名、実際は`-n 10`）、`fine_combined_8_14_14.log`、`fine_combined_8_16_16_n10.log`、`fine_direct_16plus4_n10.log`。

メモ: `README.md` の前段にある従来のT1/T2/T3最適化結果（29.10 s/plot）はこのFine Bucket高速化より前の記録。ここでの**最新のベンチ最速値は27.27 s/plot**。

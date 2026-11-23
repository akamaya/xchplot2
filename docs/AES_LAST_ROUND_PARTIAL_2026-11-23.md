# 実験A：AES最終ラウンド部分計算（2026-11-23）

## 結論

**採用。** T1/T2/T3のSYCL Compact / Tezcan経路において、AES最終ラウンドで実際に必要な32bitワードだけを計算することで、k28 / Strength 6の定常Plot生成が **25.53 → 25.18秒/Plot（0.35秒短縮、1.37%高速化）** した。生成Plotの `verify --full --trials 100` は **90/90 full proofs PASS**、終了コード0。

この記録はウォームアップ1 Plotを除いた **10 Plot測定** とFull Proof検証に基づく。

## 対象ソースと環境

- リポジトリ：`akamaya/xchplot2`、起点コミット `b6548f728f42227c83d066a7e47e66a5cae715bb`（2026-11-23 JST時点のmain）。
- OS：Windows 11 Pro。GPU：NVIDIA GeForce RTX 4070 SUPER 12GB（`gpu0`）。バックエンド：SYCL / AdaptiveCpp（CUDA）。
- Plot：`k=28`、`strength=6`、1 worker。ベンチ：1 Plotウォームアップ後、10 Plotを計測。
- T1：Compact ON、Direct24 ON、Fine Bits 18（完全展開インデックス構築用）。T2/T3：Compact/Tezcan ON、Direct20 ON、Fine Bits 16、two-phase OFF。
- VRAM削減：`POS2GPU_MATCH_INDEX_ALIAS=1`、`POS2GPU_T2_XBITS_REUSE=1`、`POS2GPU_XS_PAIR_A_ALIAS=1`、`POS2GPU_COMPACT_D=1`。全展開インデックス下でMatch Index AliasはT1/T2/T3とも成立せず（`alias T1=0 T2=0 T3=0`）；これは両測定に共通。
- 実験B（初回ラウンド定数畳み込み）：`POS2GPU_T1_FIRST_ROUND_FOLD=0` に固定。
- 実験A：`POS2GPU_AES_LAST_ROUND_PARTIAL=0` が基準、`1` が新実装。

## 実装

最終AESENCラウンドだけを、必要な出力ワードに対するT0参照4回/XOR/回転に置換する。最終ラウンドより前は**すべて通常どおり**実行する。AESENC相当のMixColumnsおよび最後のround key XORも維持する。

| 処理 | 部分計算する出力 | 備考 |
|---|---|---|
| T1 `matching_target` | `w[0]` | 検索ターゲットのみ必要 |
| T1 `pairing` | `w[3]` → 判定合格時のみ `w[0]` | 出力は合格ペアの `match_info` のみ |
| T2 `matching_target` | `w[0]` | Direct20の検索キーに使用 |
| T2 `pairing` | `w[3]` → 判定合格時のみ `w[0..2]` | `match_info` と次テーブル用metaを保持 |
| T3 `matching_target` | `w[0]` | Direct20の検索キーに使用 |
| T3 `pairing` | `w[3]` | 判定以外のAES出力は使わない。出力proof fragmentのFeistelはそのまま |

T1のStrength 6は16×16組×2＝512 AESラウンド。T2/T3は16組×2＝32 AESラウンド。1ラウンド分の省略なのでT1の相対効果は小さく、T2/T3への効果を主に期待した。T2/T3の非Tezcan・非Compact経路、およびCUDA経路には変更を加えない。

変更ファイル：`src/gpu/T1OffsetsSycl.cpp`、`src/gpu/T2OffsetsSycl.cpp`、`src/gpu/T3OffsetsSycl.cpp`、`src/gpu/AesLastRoundSycl.hpp`、`tools/parity/aes_last_round_test.cpp`、`CMakeLists.txt`、本書。

## ベンチマーク結果

**前回取得済みの基準測定（実験B OFF・実験Aなし）と今回の実験A ONを比較。** いずれもウォームアップ1 Plotを除く10 Plotの計測値。基準を取り直す必要はない。

| 指標 | 基準（OFF） | 実験A ON | ON − 基準 | 改善率 |
|---|---:|---:|---:|---:|
| Plot steady-state | 25.53秒/Plot | **25.18秒/Plot** | **−0.35秒** | **1.37%** |
| T1 match（10 Plot平均） | 7,732.72ms | 7,718.33ms | −14.39ms | 0.19% |
| T2 match（10 Plot平均） | 8,149.70ms | 7,985.42ms | **−164.28ms** | **2.02%** |
| T3 match + Feistel（10 Plot平均） | 8,442.04ms | 8,253.45ms | **−188.59ms** | **2.23%** |
| phase-timing total（10 Plot平均） | 25,524.27ms | 25,155.02ms | **−369.25ms** | **1.45%** |

備考：`steady-state` と `phase-timing total` は測定対象・集計方法が完全には同じではないため、差分を無理に一致させない。フェーズ平均は10測定Plotの値から算術平均した。計測はRTX 4070 SUPER単機、`EXITCODE=0`。

## 正当性確認

- 新たに追加した `aes_last_round_test`：AES最終ワードの演算を独立したAES参照実装と10,000ケース照合（ソース作成時のCPU側テスト）。GPU上の最終的な正当性は以下のFull Proofで確認。
- 実験A ON、k28/s6、RTX 4070 SUPERで `.gplot` を1個生成し保持。
- 実ファイル：`\\NAS-UGREEN\personal_folder\plots_test\aes_exp_a_verify\bench-7176c9e354dc1368c8bf1205330154510ce3d082cefe27829c51a963011d0030.gplot`
- `xchplot2 verify <plot> --full --trials 100`：100 trials、quality chainあり46試行、90 quality chains、**90 full proofs validated / OK / EXITCODE=0**。
- Plot生成・ベンチのいずれも終了コード0。検証した範囲では生成PlotのProof整合性に問題なし。これは全入力の形式的証明を意味するものではない。

## フラグと採用判断

- `POS2GPU_AES_LAST_ROUND_PARTIAL=1`：最終ラウンド部分計算を有効化。
- `POS2GPU_AES_LAST_ROUND_PARTIAL=0` または未設定：従来のAES計算。
- `POS2GPU_T1_FIRST_ROUND_FOLD=0`：実験Bとは独立に評価するためOFF。実装はA/B同時ONを拒否。
- 実験A：**採用**。確認された主な効果はT2/T3（合計約353ms短縮）。
- 実験B：T1だけの初回ラウンド最適化では明確な効果を確認できず、採用対象外。本実験Aのコミットには実験Bの変更を含めない。

## 残課題

- 最新の最速構成に組み込み、今後の最適化（T2/T3 AESループ展開の係数4・8・16など）の基準にする。
- 既存の他バックエンドや稀なpair-cap超過フォールバック経路は別途必要に応じて回帰検証する。

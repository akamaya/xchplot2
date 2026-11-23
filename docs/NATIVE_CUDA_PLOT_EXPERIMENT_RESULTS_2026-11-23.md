# CUDA-only k28/s6 Plot生成・Writer実験結果 — 2026-11-23

> RTX 4070 SUPER 12 GiB、Windows 11 Pro。記載した数値はユーザーが実機で提示したコンソールログの実測値。**単発のPlot完了時間、GPU計算時間、連続生成の平均スループットを混同しない。**
>
> 対象ブランチ: `cuda/nvidia-12gb-k28s6`。この実験時点で `main` にはマージしていない。

## 1. 結論・採否

- **採用（機能）:** RTX 4070 SUPER上でnative CUDA Xs→T1→T2→T3→ソート→CPU FSE→NASへの `.gplot` 出力が動作。独立した `verify --full --trials 100` が2つの単発PlotでPASS。T3の約2 GiBをCUDA Pinnedメモリに直接転送する経路と、CUDAプールを共有する2スロットのGPU producer／CPU writer consumerを実装・3枚で動作確認した。
- **採用（性能）:** GPU計算＋Pinned D2Hは **24.9937秒**（単発）、**25.126 / 25.293 / 25.049秒**（連続3枚）。連続生成でWriterとの重なりを確認した。
- **速度の結論は保留:** Pinned直接転送の単発Plot全体は **40.443秒**。元の版でユーザーが経験した単発 **26～27秒**より明確に遅い。Writerの `fflush`＋`_commit`＋`fclose` 合計が **11.105秒**かかったことが主因。ただしこの3処理の個別時間は**未測定**。連続3枚ではWriterは5.579～5.843秒だったため、15.159秒のWriterは再現性未確認の遅延として扱う。
- **実験結果:** 3枚連続の総時間 **81.414秒／3枚 = 27.138秒/Plot**（起動準備・最終Writer排出込み）。元のハイブリッド版 **25.13秒/Plot** は10枚の定常スループットであり、条件が異なるので直接の性能回帰率は算出しない。
- **安全性・前提:** CUDA新経路は秘密鍵／`--memo`を禁止、112バイトのゼロダミーメモを使用。生成時の全件SHA-256比較・重複検査は廃止し、生成後の検証のみ別実行。既存 `main` のPlot経路は変更しない。

## 2. 実験環境・再現条件

| 項目 | 設定 |
|---|---|
| 実施日 | 2026-11-23、日本時間（実行ログに時刻表示はなし） |
| Git | `akamaya/xchplot2`、`cuda/nvidia-12gb-k28s6` |
| GPU | NVIDIA RTX 4070 SUPER 12 GiB、CUDA ordinal=0、SM 8.9 |
| OS・コンパイラ | Windows 11 Pro、VS2022 Developer PowerShell、nvcc/CUDA 12.9 環境 |
| GPU構成 | k=28、strength=6、CUDA-only Candidate D、計画VRAM 8.172 GiB |
| マッチング | T1 Direct24、T2/T3 Direct20、native CUDA／CUB |
| CPU | 既存の `write_plot_file_parallel()` FSE Writer、ログ上 `threads=32` |
| Plot ID | 32バイトのゼロGroup IDからインデックス0／meta group0として導出：`0d5535e13cc9708d0ff0289af2fae27e564b6bcbcd9242f5140d96957744a517` |
| NAS出力先 | `\\NAS-UGREEN\personal_folder\plots_test\cuda_native_verify\stage4` |
| ビルド | `cmake --build .\build\windows-cuda-core --target xchplot2 -j 16` |
| 計測 | GPU: `[cuda-phase]`、Writer: `[writer-phase]`（`POS2GPU_WRITER_PHASES=1`）、Plot完了: `[plot-timing]`、連続生成: `[cuda-batch]` |
| 測定回数 | 各単発1回、連続3枚1回。NAS負荷、回線使用状況、クロック／温度は未記録 |

## 3. 実測値：単発の完全なPlot生成

| 条件・生成ファイル | GPU計算＋転送 | Writer | 完成まで | Full Proof |
|---|---:|---:|---:|---|
| CUDA-only、ハッシュ診断付きの初期版 `stage4_native_cuda.gplot` | 約27.36秒 | 未計測 | 未計測 | **109/109 PASS**（100 trials） |
| CUDA-only、SHA比較なし・通常メモリ経由 `stage4_native_cuda_speed_v2.gplot` | **25.4249秒** | **5.472秒** | **30.974秒** | 未実施／結果未提示 |
| CUDA-only、Pinned直接転送 `stage4_native_cuda_pinned_v3.gplot` | **24.9937秒** | **15.159秒** | **40.443秒** | **81/81 PASS**（100 trials） |
| 元のハイブリッド単発 | 未取得 | 未取得 | **26～27秒程度**（ユーザー申告） | 今回の同条件実測なし |

- 通常メモリ版からPinned直接転送版へ変更すると、GPU計算＋転送は **25.4249 → 24.9937秒（約0.431秒改善）**。ただしNAS書き込み同期の長い回に当たり、Plot全体は **30.974 → 40.443秒**となった。この1回差からPinned化がWriterを遅くしたとは断定できない。
- Pinned直接転送時のT3件数: **268,596,076**。T1 **268,455,895**、T2 **268,506,769**。生成ファイルは **956.64 MiB**（Writerの出力サイズ表示）。
- Pinned直接転送実行ではハッシュ診断は行っていない。正当性は生成**後**にFull Proof検証している。
- `full_plot` はGPU計算とWriter時間の単純和に加え、Plot入口の準備などを含む実測Wall時間。

### 3.1 Pinned単発のCUDAフェーズ（秒換算）

| 処理 | 時間 |
|---|---:|
| Xs生成＋ソート＋Pack | 0.255314秒 |
| T1 index＋match | 7.532399秒 |
| T1 sort | 0.078901秒 |
| T2 index＋match | 7.950424秒 |
| T2 sort＋gather | 0.209582秒 |
| T3 index＋match＋Feistel | 8.238571秒 |
| T3 sort | 0.076446秒 |
| Pinned D2H | 0.652109秒 |
| `[cuda-native-test] compute+capture` | **24.9937秒** |

`[cuda-phase] pool=0.000 ms` はフェーズ測定開始がプール構築後であるためで、VRAM確保が0秒だったという意味ではない。

### 3.2 Pinned単発のWriterフェーズ

| Writer内部処理 | 時間 |
|---|---:|
| T3境界計算 | 260.547 ms |
| FSE圧縮 | **130.598 ms** |
| `fwrite` 系書き込み | **3632.401 ms** |
| `fflush` + Windows `_commit` + `fclose` | **11105.089 ms** |
| 最終ファイル公開 | 5.238 ms |
| 合計 | **15133.873 ms** |

**重要:** 11.105秒を「`_commit`だけ」と断言してはいけない。この計測版では `fflush`・`_commit`・`fclose` の合算だった。新しいフェーズ計測コードはこの3項目を分けて表示できるが、**実測結果はまだない**。Writerの遅延はFSE圧縮ではなく書き込み・永続化側に集中した。

## 4. 実測値：3枚連続生成（GPU/FSE並列）

コマンド `test ('0'*64) --native-cuda --count 3 ... -N stage4_native_series_v3.gplot`。
Group IDを各Plotで変え、1枚ごとに異なるPlot IDを使用。2つのPinned出力スロットを交互利用し、GPUとWriterを別スレッドで並行実行した。

| Plot | T1ペア | T2ペア | T3フラグメント | GPU計算＋転送 | Writer |
|---|---:|---:|---:|---:|---:|
| `stage4_native_series_v3_0.gplot` | 268,455,895 | 268,506,769 | 268,596,076 | 25.126秒 | 5.579秒 |
| `stage4_native_series_v3_1.gplot` | 268,428,428 | 268,438,857 | 268,435,695 | 25.293秒 | 5.587秒 |
| `stage4_native_series_v3_2.gplot` | 268,463,489 | 268,495,160 | 268,534,723 | 25.049秒 | 5.843秒 |

**合計Wall: 81.414秒、3枚を含む平均27.138秒/Plot。**
GPU処理時間の合計は75.468秒で、Writerの実行がGPU処理と重なっていることを確認できる。最後のWriter排出および起動準備がWall時間へ加わる。

この結果はウォームアップ除外のsteady-state測定ではない。旧版の連続10枚 **25.13秒/Plot** と単純比較しない。今回の3枚のファイルに対する個別 `verify --full` は**未実施**（単発Pinnedファイルは81件PASS）。

## 5. 失敗・回避策と技術的考察

1. **初期CUDA-only版で不要なSHA-256処理:** T3約2 GiBのSHA-256を2回実行して約 **1.773秒**、CPU重複走査に約 **0.139秒**使用していた。診断用処理でありPlot生成には不要だったため、生成経路から撤去。正当性は生成完了後のFull Proof検証に分離した。
2. **Windows SMB保存のACLエラー:** 初期に `private storage must be owned by this user with a protected owner-only DACL` で保存失敗。ゼロ埋めダミーメモのテストPlotを機密メモと同じ扱いにしていたことが原因。ゼロメモのテストPlotだけSMB共有側ACLによる保存を許容し、秘密鍵入り経路の保護は維持した。
3. **通常メモリへの余分な約2 GiBコピー:** 初期ブリッジが小さなPinnedタイルから通常メモリ配列へ転送していた。既存の高速版に合わせ、WriterへPinned T3の`span`を直接貸与する方式を実装。単発のGPU＋転送時間は約0.43秒改善した。
4. **GPUとWriterの直列実行:** 元の `BatchPlotter` の2スロットproducer/consumer構造を、CUDA-onlyテスト連続生成に再現。3枚実測で重なりを確認できた。
5. **SMB永続化の断続的遅延:** Pinned単発のWriterは15.159秒だった一方、同じNASフォルダへの連続3枚は5.579～5.843秒。今回の測定だけではNASキャッシュ、`fflush`、`_commit`、負荷などのどれが原因か断定できない。**保存同期を無効化して数値だけ改善する変更は行わない。**

## 6. 正当性と未確認事項

- `stage4_native_cuda.gplot`: `verify --full --trials 100` → **100 trials／109 full proofs validated／OK**。
- `stage4_native_cuda_pinned_v3.gplot`: `verify --full --trials 100` → **100 trials／81 full proofs validated／OK**。
- `stage4_native_cuda_speed_v2.gplot` および `stage4_native_series_v3_{0,1,2}.gplot` のFull Proof: **未実施／ログ未提示**。
- **生成速度に検証時間を含めない。** 計測・検証の対象を明確に分離する。
- Nsight、SASS、Occupancy、CPU/NASの同時性能カウンターは**未取得**。
- この記録の後に追加した `fflush`、`_commit`、`fclose` 分離タイマーのWindows実機結果は**未取得**。

## 7. 次回の再現・確認手順（VS2022 Developer PowerShell、各1行）

現在の開発ブランチをビルドする：

```powershell
cd C:\dev\xchplot2; git pull --ff-only origin cuda/nvidia-12gb-k28s6; if ($LASTEXITCODE -ne 0) { throw 'Git更新失敗' }; cmake --build .\build\windows-cuda-core --target xchplot2 -j 16; if ($LASTEXITCODE -ne 0) { throw 'ビルド失敗' }
```

同一NASで新しいWriter分離計測を行う（新しいファイル名、機密メモなし）：

```powershell
cd C:\dev\xchplot2; $dir='\\NAS-UGREEN\personal_folder\plots_test\cuda_native_verify\stage4'; $env:POS2GPU_WRITER_PHASES='1'; try { .\build\windows-cuda-core\tools\xchplot2\xchplot2.exe test ('0'*64) --native-cuda -o $dir -N stage4_native_cuda_sync_profile_v4.gplot; if ($LASTEXITCODE -ne 0) { throw 'Plot生成失敗' } } finally { Remove-Item Env:\POS2GPU_WRITER_PHASES -ErrorAction SilentlyContinue }
```

元のハイブリッド経路（`-G`）を同じWriter計測設定で動かし、`fflush`、`fsync_commit`、`fclose` を比較する。既存ファイルは上書きしない。

```powershell
cd C:\dev\xchplot2; $dir='\\NAS-UGREEN\personal_folder\plots_test\cuda_native_verify\stage4'; $env:POS2GPU_WRITER_PHASES='1'; $sw=[System.Diagnostics.Stopwatch]::StartNew(); try { .\build\windows-cuda-core\tools\xchplot2\xchplot2.exe test ('0'*64) -G -o $dir -N stage4_hybrid_writer_baseline_v4.gplot; if ($LASTEXITCODE -ne 0) { throw '従来版生成失敗' } } finally { $sw.Stop(); Remove-Item Env:\POS2GPU_WRITER_PHASES -ErrorAction SilentlyContinue; Write-Host ("[hybrid] full_plot={0:N3}s" -f $sw.Elapsed.TotalSeconds) }
```

**備考:** 出力先はユーザー指定のNASまたは `C:\dev\xchplot2` に限る。今まで成功したPlotや結果は削除・上書きしない。

## 8. 関連資料

- [CUDA移植と検証の履歴](NATIVE_CUDA_CORE_PORT_2026-11-23.md)
- [実験記録のテンプレート](EXPERIMENT_RECORD_TEMPLATE.md)
- ソース: `src/gpu/NativeCudaStandalonePipeline.cpp`、`src/gpu/NativeCudaPool.cpp`、`src/host/GpuPlotter.cpp`、`src/host/PlotFileWriterParallel.cpp`

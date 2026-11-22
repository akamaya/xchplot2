# Candidate C: share Xs scratch with pair_a and compact pair_b (2026-11-23)

## Base

Apply on **fine-t2-xbits-reuse-20261123**, commit 0808941, containing
32-bit Fine offsets and Candidate A/B. New experimental branch:
`fine-xs-pair-a-alias-20261123`.

## Important discovery

Xs's `keys_a` split into `d_storage` was **already implemented** before C.
Merely moving or shrinking `keys_a` again would save nothing. Also, simply
shrinking `d_pair_b` to `8*cap` would corrupt B's unsorted T2 xbits at
`d_pair_b + 8*cap`. Therefore Candidate C necessarily changes T2's live
layout as well as Xs scratch placement.

## C layout (only if B ON, and Xs scratch fits pair_a)

| Phase | Allocation | Contents | Overlap rule |
| --- | --- | --- | --- |
| Xs | pair_a `0..xs_temp_bytes` | Xs scratch (3*N*u32 + sort scratch) | pair_a has no other live data |
| Xs | storage `N*8..N*12` | split `keys_a` | Xs output uses storage `0..N*8` |
| T1 match | pair_a `0..12*cap` | T1 meta/mi | Xs scratch has finished; explicit queue wait |
| T1 sort | pair_b `0..8*cap` | sorted T1 meta | pair_b unused during Xs |
| T2 match | pair_a `0..8*cap` | unsorted T2 meta | T1 output no longer live |
| T2 match | pair_a `8*cap..12*cap` | unsorted T2 xbits | no B tail in pair_b needed |
| T2 match | storage `4*cap..8*cap` | T2 match_info | storage slot 0 is sorted T1 keys |
| T2 match | storage `8*cap..12*cap` | Candidate A Fine scratch | disjoint from T2 mi |
| T2 sort | pair_b `0..4*cap` | temporary identity indices | T1 sorted meta dead after T2 match |
| T2 sort | storage slot 0/1/2 | sorted keys / source T2 mi / sorted indices | keys_in slot 1 reused later |
| T2 permute | pair_b `0..8*cap` | sorted T2 meta | identity indices dead after sort |
| T2 permute | storage `4*cap..8*cap` | sorted T2 xbits | source T2 mi dead after sort |
| T3 match | pair_a `0..8*cap` | T3 output | T2 source meta in pair_b, xbits in storage |
| T3 match | storage `8*cap..12*cap` | Candidate A Fine scratch | disjoint from T2 keys and xbits |
| T3 sort | pair_b `0..8*cap` | final fragments | sorted T2 meta dead after T3 match |

This layout adds **no GPU kernels, copies, or sorts**. It adds an explicit
`q.wait_and_throw()` after Xs when C is active so that T1 cannot overwrite
Xs scratch in `pair_a` while an out-of-order SYCL queue is still using it.
With C enabled the pool retains `pair_a` between batch plots (freed with the
pool), to avoid a new allocation before every Xs launch. With C disabled
the old per-plot `release_pair_a()` policy remains unchanged.

For **k28**: `cap=272629760`. `pair_b` is reduced from `12*cap` to
`8*cap`, saving **1090519040 bytes = 1040 MiB**, *provided* runtime Xs
scratch size fits inside `pair_a` (`12*cap`). When it does not fit,
C falls back to the legacy B layout automatically; the `[pool]` line
reports whether C is actually active and why.

`POS2GPU_XS_PAIR_A_ALIAS=0` disables C. Unset or `=1` requests C.
C is intentionally disabled when B (`POS2GPU_T2_XBITS_REUSE`) is OFF.
`POS2GPU_MATCH_INDEX_ALIAS=1` enables A. The switch is frozen per
pool/plot; do not change environment values during a running batch.

## Verification

1. Host-only C test plus existing Candidate A and B tests.
2. Windows SYCL GPU build of target `xchplot2` only.
3. Smoke: warmup=1, measured=1, keep both plots; verify each with
   `verify <plot> --full --trials 100`.
4. 10-run A/B: Candidate A and B ON in both arms; vary only C=0/1,
   setting all environment variables explicitly in each arm.
5. Check `[pool] Xs pair_a alias=1`, `pair_b=2181038080 bytes` for k28,
   successful **pool** selection, all full proofs PASS, and steady-state
   within measurement noise. If pool falls back, no VRAM saving applies.

**GPU validation and A/B benchmarking completed; results below.**

No script deletes or resets files, build folders, toolchains or user data;
no commit, merge or push is performed by applying the patch.

## Measured GPU results (2026-11-23)

- GPU: RTX 4070 SUPER, k28, strength 6; Candidates A and B enabled.
- C OFF: 26.20 s/plot; C ON: 26.15 s/plot (10 measured plots each, 1 warmup).
- GPU phase total: C OFF 26097.70 ms; C ON 26110.34 ms. Performance effectively unchanged.
- Candidate C saves 1090519040 bytes (1040 MiB) of allocated GPU pool capacity.
- Candidates A+B+C cumulatively save 2331.72 MiB (about 2.28 GiB). Not a measured peak VRAM reduction.
- Full proof verification: 236/236 PASS across 2 generated plots.
- Host tests: xs_pair_a_alias_test, t2_xbits_reuse_test, match_scratch_alias_test all PASS.
- Conclusion: Adopt Candidate C.

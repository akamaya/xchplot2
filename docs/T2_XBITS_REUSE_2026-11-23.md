# Candidate B: T2 xbits relocation within the existing GPU pool (2026-11-23)

## Goal

Reduce **allocated GPU-pool VRAM** by `cap * 4` bytes (~1040 MiB at
k28) without additional kernels, sort passes, or PCIe transfers. This patch
**requires Candidate A** (Fine match-index scratch aliases `d_storage`).
Only the pooled GPU pipeline changes; streaming paths remain unchanged.

## Data lifetime / relocation

The existing `d_pair_a` was sized for T2's 16-byte SoA output:
`meta[cap] (8 B) + match_info[cap] (4 B) + xbits[cap] (4 B)`.
T1 needs 12 bytes/entry and T3 needs 8. Candidate B reduces `d_pair_a`
to `cap*12`, removing its last `cap*4` slice. Instead:

| Phase | Source/destination | New layout | Why safe |
| --- | --- | --- | --- |
| T2 Match | `d_t2_xbits` (output) | `d_pair_b + cap*8` | Sorted T1 meta occupies only `d_pair_b[0..cap*8)` |
| T2 Sort/permute | `d_t2_xbits_sorted` (output) | `d_storage + cap*4` (`d_vals_in`) | After radix sort, `vals_in` is dead; `d_vals_out` supplies gather indices |
| T3 Match | sorted xbits (input) | `d_storage + cap*4` | No in-place sorting; source remains live |
| T3 Match Fine scratch (Candidate A) | match index | `d_storage + cap*8` | `d_keys_out` and sorted xbits remain live in preceding slots |

The T2 match Fine scratch can still borrow `d_storage + cap*4`, since no
sorted-xbits data has been placed there yet. T3 differs: Candidate A's
scratch starts after the **second** cap-sized u32 slot under B. The existing
`MatchScratchAlias.hpp` sizing helper is extended to accept distinct T2/T3
regions, and the legacy overload remains available.

Pool admission and debug report the smaller `pair_a`. `POS2GPU_T2_XBITS_REUSE`
controls this option: unset / `=1` enables; `=0` disables. Keep this value
constant within a process; the pool captures it at construction.
`POS2GPU_MATCH_INDEX_ALIAS=0` independently disables Candidate A. For the
same-binary A/B measurement, use `POS2GPU_MATCH_INDEX_ALIAS=1` in **both**
legs, and vary **only** `POS2GPU_T2_XBITS_REUSE=0/1`.

## Expected allocation savings (NOT a measured peak)

At k28, cap = 272,629,760, so `cap * 4 = 1,090,519,040 bytes = 1,040 MiB`.
With Fine16 Direct (32-bit) and Candidate A enabled, its separate sort
scratch should remain at ~68.28 MiB, because the new T3 Fine scratch
has 1,040 MiB of idle suffix space, more than the ~320 MiB required.
Thus the expected net pool-allocation reduction is ~1,040 MiB beyond A's
251.72 MiB. Actual VRAM watchdog peak and steady-state throughput require
user-side GPU measurements; no such results are claimed here.

## Verification before accepting

1. Host test: `t2_xbits_reuse_test`; existing `match_scratch_alias_test`.
2. GPU build: target `xchplot2` only, preserving the AdaptiveCpp toolchain.
3. Run a short benchmark with B enabled, keep output, verify both plots
   (`verify <file> --full --trials 100`).
4. Compare A+B OFF/ON (B=0 versus B=1, **A=1 in both**) with 10 measured
   plots and one warmup, all prior environment settings explicit.
5. Check `[pool] T2 xbits reuse` and `[pool] match-index bytes`, confirm the
   **pool** tier and no fallback, and inspect `[bench] steady-state`.
6. Reject or revise B if a repeatable performance regression appears.

No Git commit, merge, push, reset, clean, or deletion of toolchain/build
folders is performed by the installer.

## A/B results (2026-11-23)

- RTX 4070 SUPER, k28, strength 6, Candidate A ON in both runs; 1 warmup + 10 measured plots each.
- B OFF: 26.25 s/plot; B ON: 26.28 s/plot (+0.03 s, +0.11%). Performance effectively unchanged.
- GPU phase total: OFF 26107.54 ms; ON 26109.44 ms.
- Additional allocated VRAM reduction: 1090519040 bytes (1040 MiB). A+B cumulative reduction: 1291.72 MiB.
- Two generated plots: 223/223 full proofs validated.
- Conclusion: Candidate B is accepted for substantial memory savings without meaningful performance regression.

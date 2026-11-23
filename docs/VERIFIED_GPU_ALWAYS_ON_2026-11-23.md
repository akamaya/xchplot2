# k28/s6 verified GPU production profile (2026-11-23)

This branch fixes generation to k=28 / strength=6 and removes performance
opt-in selectors. All format fields remain intact for reader compatibility.

Verified paths: T1/T2/T3 local compact matching; Tezcan; AES partial final
round (Experiment A); T1 Fine18/Direct24, T2/T3 Fine16/Direct20;
Candidate B/C/D memory reuse (with size/lifetime checks).

Retired: Candidate A Match Index Alias (no saved bytes in 8,368 MiB
Direct24/Direct20 configuration), its isolated test and associated env flag.
The older 16+4 direct lookup is no longer selected for the fixed profile.

Non-default target shapes have guarded index fallback for internal testing;
streaming / other backends keep necessary safety and data-format support.

This transformation has NOT been run on the user's Windows RTX 4070 SUPER.
Build, full-proof verify, and 10-plot bench must pass BEFORE committing or
pushing. Do not use performance numbers from past binaries to validate it.

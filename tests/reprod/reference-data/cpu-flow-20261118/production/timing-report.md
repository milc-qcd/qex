# CPU source benchmark: detailed timing report

Completed 2026-11-18T00:00:24.828848-06:00 (America/Chicago). Slurm job **3732213**. Both runs passed numerical and timing validation.

## Setup

- Gauge: 0.09 fm, configuration 4800, 64³ × 96; strange mass 0.03636; flow mass 0.
- Eight SPR nodes, 256 MPI ranks (4 × 4 × 4 × 4), two OpenMP threads per rank, SIMD [2,2,1,2].
- Both modes use the same QEX-built CPU executable: MPI ICX, `-Ofast -march=sapphirerapids`, release/danger, VLEN=8.
- Wilson gauge flow and directional HISQ fermion flow: step 0.02 to 5, then 0.1 to 8; 280 steps, 81 correlator measurements.
- Corner wall: seed 425588683, time 43, corner 2; one placement, three color-diluted RHSs.
- Random wall: seed 457046288, time 16; one full-color Z4 RHS. Source QIO save/readback is retained from the earlier setup.
- Run order: corner then random wall, sequentially in the same allocation.
- Nodes: `c512-093,c516-[023,042],c517-052,c518-001,c519-031,c520-[113,131]`.

The physical source placements and run settings match job 3727297. The current unified driver measures the same six PP/A0P channels in both modes. The older corner run measured twelve meson channels, so its old total runtime is not a comparison with identical contraction work.

## End-to-end timing

| Metric | Corner wall (s) | Random wall (s) |
| --- | ---: | ---: |
| QEX application (Init–Finalize) | 2333.838 | 1926.925 |
| Launcher elapsed | 2335.875 | 1931.985 |
| Instrumented total, rank mean | 2147.625 | 1925.022 |
| Instrumented total, rank maximum | 2147.635 | 1925.031 |

## Exclusive phase breakdown

Times are accumulated wall-clock seconds, including existing communication. Percentages use the instrumented rank-mean total. Phase means add to the total; phase maxima can come from different ranks and must not be summed. No per-phase barriers were added.

| Phase | Corner mean (s) | Corner share | Corner max (s) | Random mean (s) | Random share | Random max (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gauge evolution | 117.316 | 5.46% | 130.278 | 114.438 | 5.94% | 125.735 |
| HISQ links for flow | 1338.289 | 62.31% | 1343.635 | 1278.709 | 66.43% | 1281.382 |
| Fermion evolution | 154.490 | 7.19% | 158.700 | 51.453 | 2.67% | 54.204 |
| Gauge observables | 329.096 | 15.32% | 331.818 | 329.225 | 17.10% | 331.871 |
| HISQ links for contractions | 124.434 | 5.79% | 124.472 | 123.376 | 6.41% | 123.415 |
| PP/A0P contractions | 3.885 | 0.18% | 3.919 | 2.161 | 0.11% | 2.196 |
| HISQ links/operator for solves | 3.297 | 0.15% | 3.319 | 3.302 | 0.17% | 3.317 |
| Dirac solves | 41.625 | 1.94% | 41.626 | 14.257 | 0.74% | 14.257 |
| Flow diagnostics | 1.178 | 0.05% | 1.516 | 0.951 | 0.05% | 1.134 |
| Source generation | 0.043 | 0.00% | 0.050 | 0.049 | 0.00% | 0.057 |
| Gauge input | 29.920 | 1.39% | 32.892 | 1.114 | 0.06% | 1.190 |
| Source write/read verification | 0.000 | 0.00% | 0.000 | 4.936 | 0.26% | 4.936 |
| Result output | 0.025 | 0.00% | 0.042 | 0.028 | 0.00% | 0.039 |
| Other setup/bookkeeping | 4.026 | 0.19% | 24.884 | 1.024 | 0.05% | 1.177 |

## Interpretation

- Shared gauge-related work (gauge evolution, all HISQ-link preparation, and gauge observables): **1912.432 s (89.05%)** for corner, **1849.050 s (96.05%)** for random wall.
- Pure gauge evolution alone: **5.46%** and **5.94%**, respectively. HISQ preparation is reported separately rather than being hidden in that figure.
- Fermion evolution per RK step: 0.551752 s for all three corner fields versus 0.183761 s for the one random-wall field.
- Solve time per RHS: 13.875073 s (corner) versus 14.256691 s (random wall).
- Contractions per measurement: 0.047966 s (corner) versus 0.026676 s (random wall), excluding HISQ preparation.
- Instrumented rank-mean totals excluding measured I/O: **2117.679 s** (corner), **1918.944 s** (random wall); random wall uses **9.38% less time**.
- This is one paired configuration in one allocation. Run order, filesystem caching, and machine variability can affect performance; the separate input-I/O numbers expose the major caching difference.

## Numerical validation

All 81 correlator measurements and 281 gauge-observable measurements were compared with the earlier benchmark. Corner PP uses the documented legacy temporal-phase conversion; thin and conserved A0P are compared directly. The tolerance is 1e-11 absolute plus 1e-8 times each reference channel peak.

| Run | Inversions | Iterations | Largest r² | Largest correlator difference | Largest gauge-observable difference |
| --- | ---: | --- | ---: | ---: | ---: |
| Corner | 3 | 669, 669, 669 | 9.594e-21 | 2.27374e-13 | 1.42109e-13 |
| Random wall | 1 | 680 | 9.355e-21 | 0 | 1.41887e-13 |

The saved random-wall binary source payload matches the previous run exactly. All expected timer segment counts and the MPI rank-mean accounting identity passed.

## Files

- `benchmark.json`: run commands, validation results and full timing statistics.
- `timings.tsv`: all phase means, ranges, shares and ratios.
- Each run directory contains `run.log`, `timing_C0p0_4800.json`, correlators, gauge observables, metadata and validation results.

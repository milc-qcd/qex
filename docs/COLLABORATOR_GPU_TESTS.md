# Compare GPU HISQ flow with the saved CPU results

The `hisq-flow-sycl` branch includes the completed CPU timing benchmark,
small gauge fixtures, numerical references, and executable comparison tests.
You do not need access to Mingwei's directories or CPU executable.

Start with the small tests, then replay the production case inside a GPU
allocation. The GPU implementation baseline is commit
`80cec59c4097a32e8086e13750a6b2f5331c1c87`.

```bash
git clone --branch hisq-flow-sycl https://github.com/milc-qcd/qex.git
cd qex
```

Build and select the intended executable using [README_GPU.md](../README_GPU.md)
and the [QUDA instructions](QUDA_SOLVER.md). For NVIDIA, load the matching CUDA
and SYCL runtime, select `ONEAPI_DEVICE_SELECTOR=cuda:gpu`, and set a writable
`QUDA_RESOURCE_PATH`. QUDA must be built for the allocated GPU architecture.
The existing H100 binary from another site is not a portable executable.

All Python commands below use Python 3.9+ and the standard library except
the optional independent FFT test, which also needs NumPy. Use ordinary
Python execution without `-O` (the older small-test scripts use assertions).

## What is included

The reference directory is
[`tests/reprod/reference-data/cpu-flow-20261118`](../tests/reprod/reference-data/cpu-flow-20261118).
Its `SHA256SUMS` covers all reference inputs and outputs.

| Test | Input | What must agree |
| --- | --- | --- |
| Small source and flow suite | Included unit and interacting `8^4` gauges; mass 0.2; flow 0 to 0.1 | Source bytes, source replay, each hit's correlators, averages, and gauge observables against the original CPU wall driver. |
| Small timing coverage | Output of that suite | All RK stages and solves are counted; rank statistics are valid; exclusive phase means sum to the total. |
| Independent single-wall test | Included unit gauge | Z4 source generation and PP/thin/conserved A0P against an independent FFT solve and RK3 reference. |
| Production replay | 0.09 fm, configuration 4800, `64^3 x 96` | All 81 correlator measurements, 281 gauge measurements, exact random-wall source payload, and full timing coverage. |

The frozen [CPU timing report](../tests/reprod/reference-data/cpu-flow-20261118/production/timing-report.md)
and [phase table](../tests/reprod/reference-data/cpu-flow-20261118/production/timings.tsv)
come from successful SPR job **3732213**, completed November 18, 2026.
The CPU run used eight nodes, 256 MPI ranks, and two OpenMP threads per rank.
Its rank-mean instrumented totals were **2147.625 s** for corner wall and
**1925.022 s** for random wall.

The four CPU source snapshots and recorded build settings are included for
inspection and rebuilding. Their QEX base is
`26b3941234dc7332ff28d100ed71516976e025a6`; the build used Nim 2.0.17,
MPI ICX, `-Ofast -march=sapphirerapids`, VLEN=8, release/danger.
Configure MPI/QMP/QIO paths for your site if rebuilding that historical CPU
driver. Its Sapphire Rapids-specific executable is not distributed.

## 1. Verify the downloaded reference data

Run from the repository root:

```bash
python3 tests/reprod/check_cpu_benchmark.py verify-reference
python3 -m unittest discover -s tests/reprod \
  -p test_cpu_benchmark_comparison.py -v
```

The second command tests the comparator offline, including deliberately
changed correlators, wrong sources, missing flow times, nonfinite values,
missing RK stages, incorrect timing totals, and truncated source data.
These are tests of the comparison tool, not GPU simulation results.

## 2. Run the small source and timing tests

Use a GPU allocation that can launch one and two MPI ranks. The existing
small-test harness invokes `mpirun -n 1` and `mpirun -n 2`, and tests one/two
OpenMP threads. Use your site's compatible MPI installation. The harness
uses a three-minute timeout for each individual small run.

Set absolute paths to your GPU executable and a **new** result directory:

```bash
GPU_EXE=/absolute/path/to/SYCL-h100-quda/hisqflow
SMALL_OUT=/absolute/path/to/new-small-gpu-results
REF="$PWD/tests/reprod/reference-data/cpu-flow-20261118"

python3 tests/reprod/check_wall_sources.py \
  --executable="$GPU_EXE" --solver=quda --expect-gpu \
  --reference-results="$REF/small/references" \
  --unit-gauge="$REF/small/fixtures/l8888f211b000m01m2m3a.1000.ildg" \
  --interacting-gauge="$REF/small/fixtures/l88m01m20m30.1.ildg" \
  --output="$SMALL_OUT"

python3 tests/reprod/check_flow_timings.py \
  --results="$SMALL_OUT" --backend=SYCL --solver=quda
```

This covers both source modes, N=4 seeded times, rank/thread replay, explicit
lists, N=3 with duplicate and boundary times `[7,7,0]`, N=2 interacting-field
cases, fixed corners, clock-seed replay, invalid input, and preservation of
existing source files. It independently reconstructs every saved Z4/corner
field. It compares the common CPU channels and each individual hit before
checking their mean. Source data must match exactly; floating-point values
use the existing small-suite tolerance `2e-10 + 2e-8*abs(reference value)`.
Every QUDA true residual must satisfy `r^2 <= 1.01e-20`.

Success produces `checks.json` and `timing-checks.json`, both with
`"status": "passed"`. There are 79 recorded source/numerical checks and 13
timing reports in this portable configuration. The optional old-corner
executable comparison is omitted because that executable is not required.
The `--expect-gpu` check in this harness targets NVIDIA.

For the independent N=1 free-field test, with NumPy installed:

```bash
python3 tests/reprod/check_randomwall_flow.py \
  --executable="$GPU_EXE" --solver=quda \
  --unit-gauge="$REF/small/fixtures/l8888f211b000m01m2m3a.1000.ildg" \
  --output=/absolute/path/to/new-single-wall-results
```

This adds an independent FFT/RK3 reference, rather than another comparison
between two QEX executables.

## 3. Replay the production CPU timing test

Obtain the exact gauge configuration through your collaboration's usual
data access. The large production gauge is not included in this repository.
The small gauge fixtures above are included.

| Input | Required value |
| --- | --- |
| Gauge filename | `l6496f211b630m001326m03636m4313a.4800.ildg` |
| Gauge bytes | `7247757944` |
| Gauge SHA-256 | `beeda4061921408126283a5d3eea7767f9af6f43a40820ddc304f39c8119f3e4` |
| Mass | 0.03636; fermion-flow mass 0 |
| Gauge action | Wilson |
| Flow schedule | Step 0.02 through time 5, then step 0.1 through time 8 |
| Corner source | N=1, seed 425588683, time `[43]`, corner 2; three inversions |
| Random wall | N=1, seed 457046288, time `[16]`; one inversion |

The test verifies the gauge SHA-256, supplies these options, and preserves
the filename because the driver extracts the mass from it. Random-wall
source saving/readback is enabled, matching the CPU benchmark; it writes
about 1.21 GB of source data. Corner source saving is disabled, also matching
the CPU benchmark.

Choose rank geometry and launcher for your GPU allocation. The full lattice
is much larger than the small tests; the required number of GPUs depends
on device memory and workspace size. Rank geometry need not match the CPU
run, but must divide the lattice and agree with the launcher. The test
expects VLEN=8. For example, `[2,2,2,2]` means **16 MPI ranks**, not a request
to the scheduler for 16 GPUs.

Prepare a plan without launching a simulation:

```bash
PRODUCTION_OUT=/absolute/path/to/new-cfg4800-gpu-results

python3 tests/reprod/check_cpu_benchmark.py prepare \
  --executable="$GPU_EXE" \
  --gauge=/path/to/l6496f211b630m001326m03636m4313a.4800.ildg \
  --output="$PRODUCTION_OUT" \
  --launcher='srun -n {ranks}' \
  --rank-geometry='[2,2,2,2]' --simd-geometry='[2,2,1,2]' \
  --threads=2 --backend=SYCL --solver=quda
```

Use `--launcher='mpirun -n {ranks}'` if appropriate at your site, or
`--launcher=ibrun` on Stampede3 with the allocation task count already set
to the rank-geometry product. Launcher arguments are parsed as an argument
list, not executed through a shell. For multiple nodes, use your site's
network MPI transport; `I_MPI_FABRICS=shm` is only for a single node.
The helper does not allocate resources or set site-specific GPU affinity.

Inspect `run-plan.json`. Then, **inside the matching allocation**, with the
same executable and runtime environment:

```bash
python3 tests/reprod/check_cpu_benchmark.py run --results="$PRODUCTION_OUT"
```

It runs corner then random wall sequentially, records executable/input
identities and launcher times, and automatically compares the outputs.
Use `prepare --case=random_wall` or `--case=corner` for a separate case.
Use a fresh output directory for retries; existing runs are not overwritten.

To recheck completed output without rerunning the simulation:

```bash
python3 tests/reprod/check_cpu_benchmark.py compare --results="$PRODUCTION_OUT"
```

The helper requires matching source metadata, correct solve counts and true
residuals, clean program finalization, and the complete flow grids. It compares
every value in the six shared channels and every gauge observable using
`abs(candidate-reference) <= 1e-11 + 1e-8 * reference_channel_peak`.
The channel peak is over **all flow times and sink times for that channel**.
These are the original CPU benchmark tolerances, not a guarantee that an
untested GPU setup passes. A mismatch is reported as a failure; do not loosen
the tolerance to conceal it.

The saved Z4 source payload must match the CPU payload SHA-256 exactly:
`05f05695c56d1bc00e8785bd21b5e0d4b1e1e6f2ff4db738ed8fac2a050bb41a`.
This hashes the binary field record, excluding QIO headers.

## 4. Read the timing comparison

`cpu-comparison.json` contains numerical differences by channel, source
checks, solver residuals, timing accounting, and both complete phase reports.
`timing-comparison.tsv` lists the CPU and candidate phase means/maxima and
their elapsed-time ratios. Both run and compare return nonzero on failure.

Use these comparisons carefully:

- CPU `hisqflowwall` measures six PP/A0P channels in both modes. GPU corner
  mode also measures its legacy meson channels and builds another HISQ link
  set for them. Extra channels are listed explicitly in the comparison.
  Corner contraction and total time ratios do not compare equal work.
- The CPU reference uses QEX solves on eight SPR nodes; the candidate uses
  your selected solver, GPUs, ranks, and host threads. Record those resources
  with the results. This is one CPU reference run per source mode, with corner
  first; input caching and run-to-run variation affect totals.
- GPU workspace setup/downloads and QUDA verification/cleanup are separate
  phases. CPU per-step work buffers are included in its flow phases.
  CPU diagnostic work also differs. Timer segment counts are checked against
  each implementation's expected work, not required to equal CPU counts.
- These are elapsed application times, including communication and existing
  device waits. They are not isolated GPU kernel times.
- Sum phase **rank means**, including `other`, to obtain total rank mean.
  Do not sum phase rank maxima. The total excludes timing-report output and
  QEX initialization/finalization outside the documented interval.

Return `checks.json`, `timing-checks.json`, optional single-wall `checks.json`,
and the production `run-plan.json`, `run-status.json`, `cpu-comparison.json`,
and `timing-comparison.tsv`. Keep per-case `run.log`, source metadata,
`timing_*.json`, and correlator/gauge outputs for diagnosis. Also record your
GPU model/count, compiler/runtime versions, MPI implementation, and Git commit.
The source QIO file can stay on your filesystem; its verified payload hash is
included in the comparison.

## Validation of this package

The exported small references were tested using the GPU driver's CPU
backend: **79 checks and 13 timing reports passed**. The production comparison
tool passed nine offline tests including fault injection, and the actual
configuration-4800 replay plan was prepared with the gauge checksum verified.
No new production GPU simulation was launched while packaging these results.
The H100 source/timing validation job 3732258 was still queued when this
package was prepared; successful GPU execution is not claimed here.

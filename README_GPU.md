# GPU gauge and fermion flow: build and run

This guide covers branch **`hisq-flow-sycl`** of `milc-qcd/qex`, using Intel
Max GPUs through SYCL Level Zero or NVIDIA H100 GPUs through SYCL CUDA.
Both targets use the same physics code and the `hisqflow` application.

The GPU performs gauge force/update, HISQ link smearing, the directional
fermion-flow Laplacian, Runge–Kutta vector updates, and halo packing. Gauge
fields, sources, and stage workspaces remain on the device between stages.
**The HISQ inversion defaults to CPU; measurements and configuration I/O run
on CPU.** An optional CUDA QUDA solver can be compiled and selected with
`--solver=quda`; see [QUDA build and validation](docs/QUDA_SOLVER.md). Its
hardware validation is pending; the validated flow-only builds remain available.
The implementation uses double precision, SU(3), zero fermion-flow mass, and
Wilson or rectangle gauge actions. See [implementation details](docs/HISQ_SYCL.md).

## 1. Get the source and prerequisites

```bash
git clone --branch hisq-flow-sycl https://github.com/milc-qcd/qex.git qex-gpu-sycl
cd qex-gpu-sycl
```

Required components:

- **Nim 2** and Nimble; this branch was tested with Nim **2.0.17**.
- A C++ SYCL compiler, matching SYCL runtime, and the appropriate GPU driver.
- MPI and its C/C++ wrappers, plus OpenMP support for host work.
- **QMP 2.5.4 and QIO 3.0.0**, built for that compiler/MPI environment.
- Python **3.9 or newer**, Bash, Make, and tar for the supplied tools.
- For H100: a CUDA toolkit containing `include/cuda.h` and `nvvm/libdevice`.
  The tested Stampede3 toolkit is CUDA 12.4.

Put `nim` and `nimble` on `PATH`, then install the Nim dependencies:

```bash
nim --version
nimble install -dy
```

The [package file](qex.nimble) lists the Nim dependencies, including `qiolite`.
The GPU build script invokes Nim directly; it does not require `configure`
or the machine-specific `build-mpi/qexconfig.nims`.

**A Git clone does not include compilers, QMP/QIO installations, binaries, or
input gauge configurations.** The QMP/QIO build helper expects source release
archives with a generated `configure` script. Obtain those archives from your
USQCD software installation, or use existing compatible library installations.
The upstream projects are [QMP](https://github.com/usqcd-software/qmp) and
[QIO](https://github.com/usqcd-software/qio). An unprepared Git source snapshot
may require Autotools bootstrapping before it can be used by this helper.

For a new dependency build, set these to the actual archive locations before
following either platform recipe:

```bash
export QMP_TARBALL=/absolute/path/to/qmp-2.5.4.tar.gz
export QIO_TARBALL=/absolute/path/to/qio-3.0.0.tar.gz
```

The existing Stampede3 checkout already has these archives in `deps-mpi/`,
which is the helper's default search location. If compatible libraries are
already installed, change the `QMP_DIR` and `QIO_DIR` assignments in the
chosen platform recipe to those installation prefixes and skip the
dependency-build command.

## 2. Build for Intel Max GPUs on Stampede3

Run from the repository root in a fresh shell. The supplied
`load-mpi-env.sh` loads `intel/24.0` and `impi/21.11`; it also adds the original
user's Nim installation to `PATH`. On another account, provide your own Nim
path. On another cluster, replace that environment setup with the local one.

```bash
source ./load-mpi-env.sh
export SYCL_CXX=icpx
export I_MPI_CXX=icpx OMPI_CXX=icpx MPICH_CXX=icpx
export I_MPI_CC=icx MPI_CC=mpicc
export SYCL_TARGET=generic
export QEX_BUILD_ROOT="$PWD/build-sycl"
export QEX_DEPS_ROOT="$PWD/deps-mpi"
export QMP_DIR="$QEX_DEPS_ROOT/qmp"
export QIO_DIR="$QEX_DEPS_ROOT/qio"
export VLEN=8 BUILD_JOBS=2

# First installation only; skip when these prefixes already contain libraries.
./scripts/build-flow-mpi-deps.sh

./scripts/build-hisq-sycl.sh CPU check_hisq_gpu
for target in check_sycl_smoke check_hisq_gpu check_gaugeflow_gpu check_device_halo check_gpu_mem hisqflow; do
  ./scripts/build-hisq-sycl.sh SYCL "$target"
done
```

The application is **`build-sycl/SYCL/hisqflow`**. `generic` selects the
non-CUDA build profile; the runtime selector below selects the Intel GPU.
`BUILD_JOBS` limits compiler parallelism, not MPI ranks at execution time.

## 3. Build for NVIDIA H100 on Stampede3

Use a fresh shell so Intel and NVIDIA build settings do not carry over.
The site oneAPI runtime used for the Intel build lacks the CUDA adapter needed
by this port. The installer below provides the pinned LLVM/DPC++ **v7.2.0**
compiler and matching runtime with `libur_adapter_cuda.so`.

```bash
source ./load-mpi-env.sh
python3 scripts/install-nvidia-sycl.py
source ./scripts/load-nvidia-sycl-env.sh
export SYCL_TARGET=h100
export VLEN=8 BUILD_JOBS=2

# First installation only; rebuild separately from the Intel compiler's QMP/QIO.
./scripts/build-flow-mpi-deps.sh

./scripts/build-hisq-sycl.sh CPU check_hisq_gpu
for target in check_cuda_device check_sycl_smoke check_hisq_gpu check_gaugeflow_gpu check_device_halo check_gpu_mem hisqflow; do
  ./scripts/build-hisq-sycl.sh SYCL "$target"
done
```

Defaults selected by the NVIDIA environment helper:

| Setting | Default |
| --- | --- |
| Compiler/runtime prefix | `../opt/llvm-sycl-v7.2.0-cuda`, relative to this repository |
| CUDA toolkit | `/home1/apps/nvidia/Linux_x86_64/24.5/cuda` |
| QMP/QIO prefixes | `deps-sycl/llvm-v7.2.0/{qmp,qio}` |
| Build root | `build-sycl/llvm-v7.2.0` |
| H100 application | `build-sycl/llvm-v7.2.0/SYCL-h100/hisqflow` |
| GPU target | `nvidia_gpu_sm_90` |

For a different installation location, pass `--prefix /absolute/path` to the
installer and export the same path as `QEX_NVIDIA_SYCL_PREFIX` before sourcing
the environment helper. Set `CUDA_PATH` similarly for another CUDA toolkit.
The helper also configures host-only OpenMP flags and removes old Intel SYCL
headers from compiler search paths. Use it at both build and run time.

Building H100 binaries does not need an H100, but execution does. On Stampede3,
[scripts/build-nvidia-sycl-stampede3.sbatch](scripts/build-nvidia-sycl-stampede3.sbatch)
is an optional batch build of the CPU checks and both H100/A100 targets. It
also runs SYCL CPU-device checks and therefore needs the site OpenCL CPU runtime.
Submit it from the repository root after preparing Nim, archives, and the
CUDA-enabled toolchain; override its account with `sbatch -A YOUR_ACCOUNT`.

For another platform, the build script accepts `NIM`, `CXX`, `SYCL_CXX`,
`QMP_DIR`, `QIO_DIR`, `QEX_BUILD_ROOT`, `VLEN`, and `BUILD_JOBS` overrides.
Use [the Perlmutter guide](docs/PERLMUTTER_SYCL.md) for A100/Cray MPI setup;
Stampede3-linked executables must be rebuilt for Perlmutter.

## 4. Validate on an allocated GPU node

Load the same environment used to build the executables, then run the command
for that platform **inside a single-node GPU allocation**, with at least two
MPI ranks and two host threads per rank available:

```bash
# Intel Max: after the Intel environment setup above.
ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls
ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_TARGET=generic SKIP_BUILD=1 \
  ./scripts/check-hisq-sycl.sh
```

```bash
# H100: after the NVIDIA environment setup above.
nvidia-smi
ONEAPI_DEVICE_SELECTOR=cuda:gpu sycl-ls
SYCL_TARGET=h100 SKIP_BUILD=1 ./scripts/check-hisq-sycl.sh
```

`SKIP_BUILD=1` uses the prebuilt CPU reference and GPU test programs. The H100
suite still compiles its small device-enumeration preflight, so the matching
SYCL compiler must remain available. It requires H100/CUDA, FP64, and device
USM before running the kernel tests.

Both suites compare gauge evolution and fermion flow with CPU calculations,
including one/two-rank halo exchanges. Check the printed `SYCL device:` names
and `PASS:` results. Logs are under `$QEX_BUILD_ROOT/validation/` for Intel and
`$QEX_BUILD_ROOT/validation-h100/` for H100, unless `QEX_VALIDATION_LOGS` is set.
For Intel, an unset selector defaults to **`opencl:cpu`** in the check script;
that is a CPU-device test, not an Intel GPU test.

The prebuilt H100 suite can also be submitted from the repository root:

```bash
sbatch -A YOUR_ACCOUNT scripts/validate-hisq-h100-stampede3.sbatch
```

## 5. Prepare flow and lattice inputs

Create these two files in the repository root for the configuration 1000
example. The lattice and rank geometry must match the input and launch size.

`flow-test.json`:

```json
{
  "C0p0": {
    "action": "Wilson",
    "time-increments": [0.02],
    "maximum-flow-times": [1.5],
    "loops": 1
  }
}
```

`lattice-015phys.json`:

```json
{
  "lattice-geometry": [32, 32, 32, 48],
  "rank-geometry": [2, 2, 4, 2],
  "simd-geometry": [2, 2, 1, 2]
}
```

The rank dimensions multiply to **32 MPI ranks**; SIMD dimensions multiply
to **VLEN=8**. Each local lattice dimension must also divide by its SIMD
dimension. Update the JSON and launcher together when changing the layout.
Use exactly one flow definition per run. `C0p0` is the output label.

The application reads:

```text
<base-filename>.<configuration>.ildg
```

For example, configuration 1000 with base filename
`/path/to/l3248f211b580m002426m06730m8447a` loads
`/path/to/l3248f211b580m002426m06730m8447a.1000.ildg`.
The current application extracts the **second `m...` mass** from the filename,
`0.06730` in this example, for the CPU inversion. The fermion-flow mass is zero.
A generic filename such as `config` does not supply the required mass fields.
`--contract-json` is currently unused; source and observable choices remain
in `src/alphas/hisqflow.nim`.

### Source position and reproducibility

The source options match the CPU `hisqflowwall` driver. Add them to the
usual gauge, lattice, flow, output, and solver arguments:

```bash
# N full-color Z4 walls: N inversions per flavor.
--source=random-wall --sources=4 --seed=12345

# N corner placements, three separate colors each: 3N inversions per flavor.
--source=corner-wall --sources=4 --seed=12345

# Explicit ordered time list, valid in either mode; N is inferred.
--source=random-wall --source-times='[0,17,63]' --seed=12345
```

Defaults are `--source=corner-wall` and `--sources=1`. Random times are drawn
uniformly **with replacement** from all `0..Nt-1` slices. Explicit lists must
be nonempty integer arrays in that range; duplicates and list order are
preserved. If `--sources` is also given, it must equal the list length.
The older `--source-time=T` remains a one-source shorthand and cannot be
combined with `--source-times`.

Corner positions are also drawn separately for each placement. The corner is
`x_parity + 2*y_parity + 4*z_parity`, in `0..7`; all three colors share it.
Use `--source-corner=C` to fix this corner for every placement. A corner wall
occupies one spatial parity class. Random walls occupy the entire slice and
reject `--source-corner`.

`--seed` is a uint32 value, including zero. If omitted, rank zero chooses a
clock seed and broadcasts it. The RngMilc6 stream indices match the CPU driver:

- Global lattice volume: rejection-sampled time draws.
- Global lattice volume plus one: independent corner draws.
- Global site index: Z4 noise, seeded once and advanced by three color draws
  for every random wall at every site, including off-wall sites.

The same seed, geometry, source mode, and overrides reproduce exact source
fields across MPI layouts and OpenMP thread counts. Increasing N preserves
the earlier sources. To replay hit k, preserve its index in a list of at
least k+1 sources; its time alone would select hit zero's noise. Replaying
an explicit time list with the same seed preserves the noise and corner draws; repeated random-wall times get fresh
noise. Floating-point solver/flow reductions can differ by roundoff across
backends. The corner RNG scheme has changed from the older interleaved
stream-zero draws to match CPU `hisqflowwall`; to replay an older corner run,
provide its recorded time and corner explicitly.

Before inversion, rank zero writes
`correlator_LABEL_CONFIGURATION.source.json` in the output directory (or
`source_LABEL_CONFIGURATION.json` in the working directory without `--output`).
It records the seed, N, ordered time/corner arrays, RNG scheme, geometry,
configuration, solver, inversion count, and normalization. Per-field records
are added after source generation. Existing source records are preserved;
choose a fresh output directory for another run.

Each source is contracted separately and aligned to
`dt = (sink time - source time + Nt) mod Nt`. Corner colors are summed before
averaging over the N placements. The usual correlator file contains the mean;
for N > 1, `correlator_LABEL_CONFIGURATION.sources.json` also contains every
placement's correlators. There are no contractions between different hits.
The PP/A0P source is P; the other corner channels retain their legacy
meson conventions. Host contractions follow each GPU flow download.
Gauge smearing is shared across placements at each measurement, and the GPU
flow kernels retain their existing equations. Fermion memory and work scale
with N for random walls and 3N for corner walls.

Corner mode retains all its existing meson/A0P channels and adds `PP_ss` and
`P-P_ss` to match the CPU wall driver. The relation is
`PP_ss(dt) = -(-1)^dt * pi_5_ss(dt)` and `P-P_ss = -PP_ss`. Corner channels
retain their legacy color-summed normalization without a spatial-volume
factor. Thin-link A0P includes the temporal antiperiodic sign once; the HISQ
links already contain their boundary signs.

### Full-color Z4 random wall

Each random wall has independent `{1, i, -1, -i}` entries in all three color
components at every spatial site of its time slice, and zero elsewhere.
`--solver=quda` performs the N inversions with QUDA; SYCL evolves all resulting
propagators together with the gauge field. The true QEX residual is checked
after every QUDA inversion before QUDA shuts down and SYCL flow begins.

Random mode outputs `PP_ss`, its negative `P-P_ss`, and real/imaginary parts of
`A0P_thin` and `cA0P`. It divides by spatial volume and sums color. A0P uses
the forward current; the conserved current includes the Naik contribution
on every elementary link crossed. Corner taste phases are not used.

In either mode, `--save-source=wall.qio` saves double-precision QIO fields and
checks exact read-back before solving. One random wall retains `wall.qio`.
Multiple random walls use `wall.source0000.qio`, etc.; corner fields use
`wall.source0000.color0.qio`, `.color1.qio`, `.color2.qio`, etc. Existing QIO
fields are preserved.

`tests/reprod/check_wall_sources.py` checks multiple placements against the
CPU wall driver, independently reconstructs saved fields, checks rank/thread
and explicit-list replay, tests free and optional interacting gauge fields,
and verifies all old corner channels with a supplied reference executable:

```bash
python3 tests/reprod/check_wall_sources.py \
  --executable="$QEX_BUILD_ROOT/CPU/hisqflow" \
  --reference-executable=/path/to/cpu/hisqflowwall \
  --unit-gauge=/path/to/l8888f211b000m01m2m3a.1000.ildg \
  --output=/path/to/new-cpu-checks
```

For H100, use the `SYCL-h100-quda/hisqflow` executable with `--solver=quda`
and `--expect-gpu`. The validated CPU references can be reused with
`--reference-results=/path/to/new-cpu-checks/cpu-reference` instead of
`--reference-executable`. Pass the same optional `--interacting-gauge` to
both runs. Run GPU checks in an allocation.

For numerical regression, `tests/reprod/check_randomwall_flow.py` checks the
source against an independent MILC6 implementation and the correlators
against a free-HISQ Fourier solve followed by RK3 flow. It covers odd/even
source times, MPI decomposition, clock-seed replay, fixed source time,
source preservation errors, and an optional corner-wall regression.
Run it in a compute allocation with NumPy and a unit `8^4` gauge fixture:

```bash
python3 tests/reprod/check_randomwall_flow.py \
  --executable="$QEX_BUILD_ROOT/CPU/hisqflow" \
  --unit-gauge=/path/to/l8888f211b000m01m2m3a.1000.ildg \
  --output=/path/to/new-check-directory
```

Use the `SYCL-h100-quda/hisqflow` executable and `--solver=quda` for the same
checks on H100. The fixture's `m2` token selects mass `0.2`. The generator is
`tests/backend/make_randomwall_unit.nim`; build it with
`QUDA_DIR= ./scripts/build-hisq-sycl.sh CPU make_randomwall_unit` and run the resulting
executable with the fixture filename as its only argument inside an allocation.

## 6. Run the full flow with Slurm on Stampede3

Save the following as `run-gpu.sbatch` in the repository root. Replace
`YOUR_ACCOUNT` and `GAUGE_STEM`. This example uses the default build directories
from sections 2 and 3, one node, 32 MPI ranks, and two host threads per rank.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=hisq-flow-gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
repo=$PWD
source ./load-mpi-env.sh
case "${GPU_TARGET:?Set GPU_TARGET=intel or h100}" in
  intel)
    export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
    binary="$repo/build-sycl/SYCL/hisqflow"
    ;;
  h100)
    source ./scripts/load-nvidia-sycl-env.sh
    export ONEAPI_DEVICE_SELECTOR=cuda:gpu
    binary="$repo/build-sycl/llvm-v7.2.0/SYCL-h100/hisqflow"
    "$repo/build-sycl/llvm-v7.2.0/SYCL-h100/check_cuda_device" H100
    ;;
  *) echo "Unknown GPU_TARGET: $GPU_TARGET" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=2 OMP_PROC_BIND=spread OMP_PLACES=cores
export I_MPI_FABRICS=shm I_MPI_PIN=1 I_MPI_PIN_DOMAIN="$SLURM_CPUS_PER_TASK"
export SLURM_CPU_BIND=cores
GAUGE_STEM=/path/to/l3248f211b580m002426m06730m8447a
out="${SCRATCH:?}/qex-flow/${GPU_TARGET}-${SLURM_JOB_ID}"
mkdir -p "$out"
sycl-ls | tee "$out/device-inventory.txt"
/usr/bin/time -v -o "$out/resource-usage.txt" \
  ibrun "$binary" \
    --flow-json="$repo/flow-test.json" \
    --lattice-json="$repo/lattice-015phys.json" \
    --configuration=1000 \
    --base-filename="$GAUGE_STEM" \
    --output="$out" 2>&1 | tee "$out/run.log"
```

Choose **one** submission command for the intended hardware:

```bash
# Intel Max GPUs:
sbatch --partition=pvc --export=ALL,GPU_TARGET=intel run-gpu.sbatch
# NVIDIA H100 GPUs:
sbatch --partition=h100 --export=ALL,GPU_TARGET=h100 run-gpu.sbatch
```

These Stampede3 queues allocate the node exclusively; these examples do not
request GPU GRES. Follow the local scheduler's resource syntax on other systems.
For an existing suitable allocation, the `ibrun` invocation and environment
above can be used directly with an explicit output directory.

QEX assigns MPI rank modulo the number of visible devices. With 32 ranks and
four visible H100s, eight ranks share each GPU. GPU memory must fit the combined
allocations of those ranks, and host RAM must fit the CPU workspaces. Peak GPU
usage and performance depend on the rank geometry and halo size.

**This example is single-node.** `I_MPI_FABRICS=shm` only supports shared-memory
communication. Multi-node use requires the site's network MPI transport,
consistent device binding/rank geometry, and an additional communication check.
Halo exchange currently stages packed boundary messages through host memory;
GPU-aware MPI is not required by this implementation.

## 7. Inspect and compare the output

For a portable test package with saved CPU results, small gauge fixtures,
and an exact replay of the 0.09 fm configuration-4800 timing benchmark, see
[the collaborator GPU test guide](docs/COLLABORATOR_GPU_TESTS.md). It provides
source/correlator comparisons and phase-timing checks without requiring the
original CPU executable or access to the author's filesystem.

For the example above, `$out` contains:

- `correlator_C0p0_1000`: correlators in JSON format.
- `gaugeobs_C0p0_1000`: gauge observables in JSON format.
- `timing_C0p0_1000.json`: phase timings with MPI rank min/mean/max seconds
  and segment counts.
- `run.log`: device selection, solver/flow messages, and application elapsed time.
- `resource-usage.txt`: GNU time statistics; its RSS is not necessarily the sum
  of all MPI ranks and does not measure GPU memory.

To compare **both** output files with a saved CPU run:

```bash
python3 scripts/compare-flow-output.py \
  --reference /absolute/path/to/cpu-results \
  --candidate /absolute/path/to/gpu-results \
  --configuration 1000 --label C0p0 \
  --rtol 1e-8 --atol 1e-10 \
  --report comparison.json --csv comparison.csv
```

The comparator requires identical keys and array lengths, rejects nonfinite
values, and checks every value with `abs(GPU-CPU) <= 1e-10 + 1e-8*abs(CPU)`.
It exits nonzero on failure and reports errors by channel and flow time.
For this example, complete output has 76 gauge times and 16 correlator times,
ending at flow time 1.50, with 48 correlator timeslices.

The specialized [Intel comparison job](scripts/compare-015phys-config1000.sbatch)
and [H100 comparison job](scripts/compare-015phys-config1000-h100.sbatch) reproduce
the recorded configuration 1000 tests. They require a prepared validation
directory with frozen binaries, inputs, reference files, checksum manifest,
and `validate_output.py`; they are not standalone launchers from a fresh clone.
The H100 comparison job additionally samples GPU memory once per second.

### Phase timings

Every `hisqflow` run writes `timing_LABEL_CONFIGURATION.json` alongside its
ordinary output, and prints a `FLOW_TIMING` summary. No extra option is needed.
The report follows the CPU `hisqflowwall` format: monotonic wall-clock seconds,
exclusive phases, and min/mean/max across MPI ranks. It records the backend,
solver, configuration, source mode/count, number of solution fields, and seed.

The shared CPU/GPU phases are `gauge_flow`, `fermion_flow`, `contractions`,
`solve`, `hisq_links_flow`, `hisq_links_solve`, `hisq_links_contractions`,
`gauge_measurements`, `flow_diagnostics`, `source_generation`, `source_io`,
`gauge_input`, and `output_io`. Flow HISQ construction covers all three RK
stages and is shared by the sources. Fermion flow and solve times include all
N random-wall or 3N corner-wall fields; contractions include source averaging.
HISQ construction is excluded from the solve, flow-arithmetic, and contraction
phases so it is not counted twice.

The accelerator driver also separates these costs:

| Phase | Work included |
| --- | --- |
| `flow_setup` | Persistent flow workspaces, halo plans, allocation, initial host copies, and cleanup. |
| `flow_download` | Explicit full-field snapshots and copies for CPU measurements. |
| `solve_verification` | QEX true-residual checks after QUDA solves. |
| `solver_cleanup` | QUDA shutdown after the solves. |

These are elapsed application times, including CPU work, communication, and
device waits. The current SYCL kernel and copy wrappers wait before returning,
and QUDA returns a completed host solution. Timers therefore include completed
device work without adding queue waits or MPI barriers. They are not isolated
GPU kernel times. Lazy uploads and halo transfers remain in the phase that
requests them; `flow_download` does not represent all data transfer.

The total starts immediately after `qexInit` and ends after ordinary output.
It excludes timing-report generation/writing and `qexFinalize`. `other`
accounts for remaining setup, bookkeeping, logging, garbage collection, and
timer overhead. Phase **rank means**, including `other`, sum to the total
rank mean. Phase maxima can belong to different ranks and must not be added
to obtain wall time. Use total rank maximum for the slowest rank's elapsed
time. CPU fallback work buffers are counted within its flow phases, while the
coupled implementation reports persistent workspace costs in `flow_setup`.

[`check_flow_timings.py`](tests/reprod/check_flow_timings.py) checks phase
coverage/counts, rank statistics, accounting, and numerical agreement with
an untimed executable on small unit and interacting gauge fields. It also
checks timing reports from a completed `check_wall_sources.py` run using
`--results=PATH --backend=SYCL --solver=quda`.

## Validation record and troubleshooting

The recorded Intel Max run passed all **10,280** gauge/correlator values for
configuration 1000. H100 small-lattice gauge, fermion-flow, and MPI checks also
passed. As of November 10, 2026, the full H100 configuration 1000 run was queued;
that full-run comparison is not yet claimed. See
[configuration 1000 results](docs/CONFIG1000_VALIDATION.md) and
[NVIDIA runtime validation](docs/NVIDIA_SYCL_RUNTIME.md).

| Symptom | Check |
| --- | --- |
| No NVIDIA device / missing CUDA adapter | Source the CUDA-enabled environment, use `cuda:gpu`, and run on an allocated NVIDIA node. |
| `_intel_fast_memcpy` unresolved when linking with LLVM | Rebuild QMP/QIO with the LLVM environment; do not reuse the Intel compiler's archives. |
| OpenMP GPU-offload error with the NVIDIA compiler | Use the helper's host-only `OMPFLAG`; a plain `-fopenmp` requests extra offload with this compiler. |
| Missing Nim package | Run `nimble install -dy` with the intended Nim/Nimble installation. |
| Layout assertion | Check rank count, lattice divisibility, and SIMD geometry against `VLEN`. |
| GPU allocation failure | Check memory on each device and the number of ranks sharing it; host RAM is separate. |

The main implementation files are
[`gaugeflowgpu.nim`](src/alphas/gaugeflowgpu.nim),
[`hisqflowgpu.nim`](src/alphas/hisqflowgpu.nim),
[`flowstepgpu.nim`](src/alphas/flowstepgpu.nim), and
[`halodevice.nim`](src/comms/halodevice.nim). Selection is wired through
[`hisqflowutils.nim`](src/alphas/hisqflowutils.nim). The build script defines
`hisqFlowGpu` and selects `Backend=SYCL`; `Backend=CPU` runs the accelerator
implementation through the host backend for tests. The original CPU flow path
is selected when the application is built without `hisqFlowGpu`.

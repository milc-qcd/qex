# GPU gauge and fermion flow: build and run

This guide covers branch **`hisq-flow-sycl`** of `milc-qcd/qex`, using Intel
Max GPUs through SYCL Level Zero or NVIDIA H100 GPUs through SYCL CUDA.
Both targets use the same physics code and the `hisqflow` application.

The GPU performs gauge force/update, HISQ link smearing, the directional
fermion-flow Laplacian, Runge–Kutta vector updates, and halo packing. Gauge
fields, sources, and stage workspaces remain on the device between stages.
**The HISQ inversion, measurements, and configuration I/O still run on CPU.**
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

For the example above, `$out` contains:

- `correlator_C0p0_1000`: correlators in JSON format.
- `gaugeobs_C0p0_1000`: gauge observables in JSON format.
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

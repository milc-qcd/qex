# Optional QUDA solver with SYCL flow

The application accepts **`--solver=qex`** (the default; `cpu` is an alias) or
**`--solver=quda`**. Both choices keep gauge and fermion flow on the selected
SYCL GPU. QUDA accelerates the initial HISQ inversion; measurements and I/O
remain on the host.

As of November 10, 2026, the QUDA library and the CPU/QUDA application and
test executables compiled and linked successfully on the login node with two
build workers. Solver-selection error checks passed. H100 numerical validation
is submitted as job **3706002**, directly from the test directory with no
build-job dependencies. A successful QUDA solve is **not yet claimed**.
The previously validated flow-only executables remain separate.

## Build on Stampede3

Use Nim, MPI, QMP/QIO, and the CUDA-enabled SYCL installation described in
[README_GPU.md](../README_GPU.md). QUDA additionally needs CMake 3.25.2 or
newer, a CUDA-capable host compiler, and Eigen headers. For H100, use CUDA
12.6 or newer with this QUDA revision; CUDA 12.8 is installed on Stampede3.
CUDA 12.4 failed to compile QUDA's Hopper tensor-memory descriptor helpers. The build helper pins
[QUDA revision af7e15080e247795ef076b6d9284478755a9a683](https://github.com/lattice/quda/tree/af7e15080e247795ef076b6d9284478755a9a683).
It builds a shared CUDA library with staggered operators, the MILC interface,
and MPI communication. QMP and GPU-aware MPI are not required inside QUDA;
QEX continues to use its existing QMP-over-MPI setup.

From a fresh shell in the repository root:

```bash
source ./load-mpi-env.sh
export CUDA_PATH=/home1/apps/nvidia/Linux_x86_64/25.3/cuda
export QEX_BUILD_ROOT="$PWD/build-sycl/llvm-v7.2.0-cuda12.8"
source ./scripts/load-nvidia-sycl-env.sh
export SYCL_TARGET=h100 BUILD_JOBS=2
export QUDA_BUILD_DIR="$PWD/deps-quda/build-sm_90-cuda12.8"
export QUDA_EIGEN_INCLUDE=/home1/apps/nvidia25/eigen/3.4.0/include/eigen3
export QUDA_CUDA_MATH_ROOT=/home1/apps/nvidia/Linux_x86_64/25.3/math_libs

# Build with two concurrent compiler processes on the login node.
nice -n 10 ./scripts/build-quda.sh
```

Compilation needs a CPU and the toolkit, not a physical GPU. `BUILD_JOBS`
limits compiler concurrency only; it does not limit runtime MPI/OpenMP/GPU
parallelism. [TACC permits compilation on login nodes with limited concurrency](https://docs.tacc.utexas.edu/basics/conduct/).
For more build workers, use a CPU allocation or the optional
`scripts/build-quda-stampede3.sbatch` builder.

The source/build/install locations in this example are `deps-quda/source`,
`deps-quda/build-sm_90-cuda12.8`, and `deps-quda/sm_90`. Without
`QUDA_BUILD_DIR`, the build directory defaults to `deps-quda/build-sm_90`.
`QEX_QUDA_ROOT`, `QUDA_DIR`, `QUDA_HOST_CC`, and `QUDA_HOST_CXX` can override
those choices. With `SYCL_TARGET=a100`, the helper uses `sm_80` and separate
build/install directories. A100/Perlmutter execution is not validated here.

The site HPC SDK separates CUDA runtime and math libraries. The helper resolves
the actual `nvcc` path and accepts `QUDA_CUDA_MATH_ROOT` for CMake discovery.
For a normal, combined CUDA installation, that override can be omitted. If
`QUDA_EIGEN_INCLUDE` is unset, QUDA downloads its configured Eigen dependency.
The source checkout must match the pinned revision and have no local changes.
The helper installs from QUDA's `lib` build directory, which includes the
production library, headers, and CMake package. This avoids an upstream
install rule that requests an unbuilt test-helper library even with tests off.

After the library build completes, load its environment and compile the QEX
application and dedicated test:

```bash
export QEX_QUDA_BUILD_RECORD="$QEX_BUILD_ROOT/quda-build-records/local-build"
nice -n 10 ./scripts/build-quda-flow.sh
source ./scripts/load-quda-env.sh
```

The new executables are:

```text
build-sycl/llvm-v7.2.0-cuda12.8/SYCL-h100-quda/hisqflow
build-sycl/llvm-v7.2.0-cuda12.8/SYCL-h100-quda/check_quda_hisq
```

Setting `QUDA_DIR` enables QUDA support and gives the binaries/caches a `-quda`
suffix. Leaving it empty produces the original flow-only build. The device
preflight remains in `SYCL-h100/check_cuda_device` because it does not use QUDA.
The build script supplies `qudaDir`, `cudaLibDir`, and `cudaMathLibDir` Nim
settings and the CUDA/SYCL interoperability definitions. `CUDA_LIB_DIR` and
`CUDA_MATH_LIB_DIR` select runtime library locations; load the same environment
when executing the result. The linker uses CUDA driver/NVML toolkit stubs on
CPU/login nodes; those stub directories are not added to the runtime search
path. The actual NVIDIA driver libraries are supplied by the GPU node.
`load-quda-env.sh` selects `QUDA_GCC_TOOLCHAIN=/opt/apps/gcc/13.2.0` so the
SYCL compiler uses the same C++ headers and runtime as the QUDA build;
otherwise clang selects system GCC 11 and cannot link QUDA's GCC 13 symbols.
Override that prefix when using a different GCC installation.

The local builder compiles the CPU reference, CUDA device preflight, QUDA
application, and dedicated test. It checks solver-selection error paths,
records executable/library hashes, and writes `exit-status.txt`. Numerical
CPU and GPU tests run in the subsequent H100 allocation. The optional batch
builder runs the same compiler helper with `--run-cpu-tests` inside an
allocation.

To submit directly from a prepared test directory after a successful build:

```bash
export QEX_REPO_ROOT="$PWD"
export QEX_QUDA_VALIDATION_ROOT=/absolute/path/to/prepared-test-directory
export QUDA_DIR="$QEX_REPO_ROOT/deps-quda/sm_90"
cd "$QEX_QUDA_VALIDATION_ROOT"
sbatch "$QEX_REPO_ROOT/scripts/validate-quda-h100-stampede3.sbatch"
```

Keep the `QEX_BUILD_ROOT`, `QEX_QUDA_BUILD_RECORD`, and CUDA environment from
the build. There are no Slurm build-job dependencies. The job verifies the
completed build record and checksums before running. Its Slurm log is written
to the submit directory as `quda-h100-validation-JOBID.log`.

## Select the solver at runtime

Inside an H100 allocation, with the NVIDIA and QUDA environments loaded:

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export QUDA_RESOURCE_PATH="$SCRATCH/qex-quda-tuning"
mkdir -p "$QUDA_RESOURCE_PATH"

# Match the MPI count to rank-geometry in the supplied lattice JSON.
ibrun "$QEX_BUILD_ROOT/SYCL-h100-quda/hisqflow" \
  --solver=quda --seed=12345 \
  --flow-json=/absolute/path/to/flow-test.json \
  --lattice-json=/absolute/path/to/lattice.json \
  --configuration=1000 \
  --base-filename=/absolute/path/to/l3248f211b580m002426m06730m8447a \
  --output=/absolute/path/to/existing-output-directory
```

Use `--solver=qex` with the same executable and inputs to select the CPU solve.
There is no recompile between these choices. Source selection is shared by
both solvers; use the same `--seed`, `--source-time`, and `--source-corner`
when comparing them. The source record and source-relative time convention
are described in [the GPU guide](../README_GPU.md#source-position-and-reproducibility).
A binary built without `QUDA_DIR`
rejects `--solver=quda` before loading the configuration. Invalid solver names
are also rejected. The application explicitly selects QEX by default, even
though QEX's general solver-parameter defaults select QUDA when it is linked.

For the initial H100 validation, use **four MPI ranks and sixteen host threads
per rank**, with lattice rank geometry `[1,1,2,2]` on `32 x 32 x 32 x 48` and
SIMD geometry `[2,2,1,2]`. This gives one rank per visible H100 on the tested
node type. The earlier 32-rank flow-only setup is not the QUDA test layout.
Recheck memory and throughput when changing that decomposition.

The environment helper sets `QUDA_ENABLE_GDR=0` and `QUDA_ENABLE_P2P=0` for the
initial host-buffer MPI validation. Enabling direct GPU communication is a
separate performance exercise. `QUDA_DEVICE_RESET=1` is rejected for the
combined build because it would destroy the CUDA context needed by SYCL.
The supplied helper sets it to zero.

## Numerical checks and implementation

The adapter packs QEX fields into the existing MILC host layout and passes
fat/Naik links to QUDA's `qudaInvert`. It explicitly sets CG, the requested
residual/iteration limit, `tadpole=1`, and `naik_epsilon=0`; the current HISQ
coefficients and antiperiodic boundary signs are already in the QEX links.
The first implementation uses host transfers at the solve boundary.

For SYCL builds, QUDA selects the native CUDA device ordinal of the actual
SYCL device. It initializes that device before the MILC initializer, which
otherwise chooses a device from MPI topology. The wrapper preserves QEX's rank
map for MPI. QUDA initialization is lazy, and cleanup is conditional and
idempotent. The application releases QUDA's workspaces after inversion and
before allocating the resident flow workspace.

After every QUDA inversion, the application recomputes the true residual with
QEX's original unpreconditioned Dirac operator. Nonfinite values or a squared
relative residual greater than `1.01 * sp.r2req` stop the run. The 1% allowance
covers the final independent residual recomputation; the solve itself retains
the original `sp.r2req=1e-20` target.

The dedicated test checks unit and random gauge fields, even/odd/full sources,
true residuals, and CPU/QUDA propagator agreement. It then runs two coupled
flow steps after QUDA cleanup, checking gauge and fermion results and released
SYCL allocations. Run on an allocated H100 node:

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:gpu OMP_NUM_THREADS=2
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork
mpiexec -n 1 "$QEX_BUILD_ROOT/SYCL-h100-quda/check_quda_hisq"
mpiexec -n 2 "$QEX_BUILD_ROOT/SYCL-h100-quda/check_quda_hisq"
```

## Validation submission

The [recorded validation status](validation/quda-local-build-20261110.json)
distinguishes completed compilation checks from pending runtime tests.

| Stage | Location / job | Status when recorded |
| --- | --- | --- |
| QUDA library and QEX application/test build | Login node, two workers | Completed |
| H100 numerical tests and full configuration 1000 comparisons | 3706002 | Pending for priority; no dependencies |

The earlier dependency chain, jobs 3705169, 3705235, and 3705247, was cancelled.
The direct submission uses the frozen `build/` payload and `run.sbatch` in the
test directory below. Its Slurm log is `quda-h100-validation-3706002.log`.

The final job first requires the small one/two-rank tests to pass. It then
runs configuration 1000 twice using the same executable and four-rank layout:
CPU inversion plus SYCL flow, followed by QUDA inversion plus SYCL flow.
Both outputs are compared against the saved CPU reference, and against each
other, with the unchanged all-value criterion
`abs(new-reference) <= 1e-10 + 1e-8*abs(reference)`. Both gauge observables and
correlators must have complete time/channel coverage. Runtime and sampled
GPU memory are recorded separately for each solver.

Prepared inputs and future results are under:

```text
/work2/11599/mdai07/stampede3/test/hisqflow_directional_20261026/gpu-sycl-validation/015phys-cfg1000-20261110-quda-h100
```

The full-validation batch script accepts this directory as its submit
directory, `QEX_REPO_ROOT` for the source-tree location, and
`QEX_QUDA_BUILD_RECORD` for the completed local build record. It validates the
build's binary checksums and the input manifest before execution. The older
`QEX_QUDA_BUILD_JOB` selector remains available for optional batch builds. New attempts go under `attempts/job_JOBID/`, with
separate `qex/` and `quda/` result directories. This does not modify the original
flow-only H100 job 3699217 or its frozen inputs.

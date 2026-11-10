# Optional QUDA solver with SYCL flow

The application accepts **`--solver=qex`** (the default; `cpu` is an alias) or
**`--solver=quda`**. Both choices keep gauge and fermion flow on the selected
SYCL GPU. QUDA accelerates the initial HISQ inversion; measurements and I/O
remain on the host.

The QUDA integration is undergoing validation. As of November 10, 2026,
the CPU application compiled and linked, and its unavailable/invalid solver
checks passed. The QUDA-enabled application and dedicated test compiled all
C++/SYCL objects for H100 with linking disabled. The QUDA library build, final
application/test linking, and NVIDIA numerical checks are queued; a successful
QUDA solve is **not yet claimed**. The previously validated
flow-only executables remain separate.

## Build on Stampede3

Use Nim, MPI, QMP/QIO, and the CUDA-enabled SYCL installation described in
[README_GPU.md](../README_GPU.md). QUDA additionally needs CMake 3.25.2 or
newer, a CUDA-capable host compiler, and Eigen headers. The build helper pins
[QUDA revision af7e15080e247795ef076b6d9284478755a9a683](https://github.com/lattice/quda/tree/af7e15080e247795ef076b6d9284478755a9a683).
It builds a shared CUDA library with staggered operators, the MILC interface,
and MPI communication. QMP and GPU-aware MPI are not required inside QUDA;
QEX continues to use its existing QMP-over-MPI setup.

From a fresh shell in the repository root:

```bash
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
export SYCL_TARGET=h100
export QUDA_EIGEN_INCLUDE=/home1/apps/nvidia25/eigen/3.4.0/include/eigen3
export QUDA_CUDA_MATH_ROOT=/home1/apps/nvidia/Linux_x86_64/24.5/math_libs

# Configure/download dependencies on the login node; compile on a CPU allocation.
./scripts/build-quda.sh --configure-only
sbatch -A YOUR_ACCOUNT scripts/build-quda-stampede3.sbatch
```

Alternatively, inside a suitable CPU allocation, run
`BUILD_JOBS=8 ./scripts/build-quda.sh`. The default source/build/install locations
are `deps-quda/source`, `deps-quda/build-sm_90`, and `deps-quda/sm_90`.
`QEX_QUDA_ROOT`, `QUDA_DIR`, `QUDA_HOST_CC`, and `QUDA_HOST_CXX` can override
those choices. With `SYCL_TARGET=a100`, the helper uses `sm_80` and separate
build/install directories. A100/Perlmutter execution is not validated here.

The site HPC SDK separates CUDA runtime and math libraries. The helper resolves
the actual `nvcc` path and accepts `QUDA_CUDA_MATH_ROOT` for CMake discovery.
For a normal, combined CUDA installation, that override can be omitted. If
`QUDA_EIGEN_INCLUDE` is unset, QUDA downloads its configured Eigen dependency.
The source checkout must match the pinned revision and have no local changes.

After the library build completes, load its environment and compile the QEX
application and dedicated test:

```bash
source ./scripts/load-quda-env.sh
./scripts/build-hisq-sycl.sh SYCL check_cuda_device
./scripts/build-hisq-sycl.sh SYCL check_quda_hisq
./scripts/build-hisq-sycl.sh SYCL hisqflow
```

The new executables are:

```text
build-sycl/llvm-v7.2.0/SYCL-h100-quda/hisqflow
build-sycl/llvm-v7.2.0/SYCL-h100-quda/check_quda_hisq
```

Setting `QUDA_DIR` enables QUDA support and gives the binaries/caches a `-quda`
suffix. Leaving it empty produces the original flow-only build. The device
preflight remains in `SYCL-h100/check_cuda_device` because it does not use QUDA.
The build script supplies `qudaDir`, `cudaLibDir`, and `cudaMathLibDir` Nim
settings and the CUDA/SYCL interoperability definitions. `CUDA_LIB_DIR` and
`CUDA_MATH_LIB_DIR` select runtime library locations; load the same environment
when executing the result.

The combined batch builder
[scripts/build-quda-flow-stampede3.sbatch](../scripts/build-quda-flow-stampede3.sbatch)
performs CPU regression/error-path checks and builds the QUDA application and
test. Submit it with `--dependency=afterok:LIBRARY_JOB_ID` after submitting the
library build. Only the library-build job needs to download/configure QUDA.

## Select the solver at runtime

Inside an H100 allocation, with the NVIDIA and QUDA environments loaded:

```bash
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export QUDA_RESOURCE_PATH="$SCRATCH/qex-quda-tuning"
mkdir -p "$QUDA_RESOURCE_PATH"

# Match the MPI count to rank-geometry in the supplied lattice JSON.
ibrun "$QEX_BUILD_ROOT/SYCL-h100-quda/hisqflow" \
  --solver=quda \
  --flow-json=/absolute/path/to/flow-test.json \
  --lattice-json=/absolute/path/to/lattice.json \
  --configuration=1000 \
  --base-filename=/absolute/path/to/l3248f211b580m002426m06730m8447a \
  --output=/absolute/path/to/existing-output-directory
```

Use `--solver=qex` with the same executable and inputs to select the CPU solve.
There is no recompile between these choices. A binary built without `QUDA_DIR`
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

## Submitted validation chain

The [recorded validation status](validation/quda-integration-20261110.json)
distinguishes completed compilation checks from pending runtime tests.

| Stage | Job | Status when recorded |
| --- | --- | --- |
| Pinned H100 QUDA library build | 3705169 | Pending for priority |
| QEX CPU regression and QUDA application/test builds | 3705235 | Depends on successful library build |
| H100 numerical tests and full configuration 1000 comparisons | 3705247 | Depends on successful QEX build |

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

The full-validation batch script requires this prepared directory via
`QEX_QUDA_VALIDATION_ROOT` and the application-build job ID via
`QEX_QUDA_BUILD_JOB`. It validates the build's binary checksums and the input
manifest before execution. New attempts go under `attempts/job_JOBID/`, with
separate `qex/` and `quda/` result directories. This does not modify the original
flow-only H100 job 3699217 or its frozen inputs.

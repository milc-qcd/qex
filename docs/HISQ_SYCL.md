# Directional HISQ flow with SYCL

This checkout is an independent copy of `milc-qex` at commit
`26b3941234dc7332ff28d100ed71516976e025a6`, including the local changes present
when it was copied. Development is on branch `hisq-flow-sycl`.

Build with `-d:hisqFlowGpu` to select the resident accelerator flow in
`src/alphas/hisqflow.nim`. `-d:Backend=SYCL` selects SYCL; `Backend=CPU`
runs the same kernels through QEX's CPU backend. Without `hisqFlowGpu`, the
application retains the original CPU flow implementation.

## Execution and source guide

The coupled RK3 integrator now keeps the gauge field, all fermion sources,
and stage temporaries on the selected device. Each of its three stages runs:

1. Staggered phases and antiperiodic temporal boundary signs on thin links.
2. First Fat7 smearing, U(3) projection, and second Fat7/Lepage plus Naik links.
3. The massless directional HISQ Laplacian and fermion RK vector combinations.
4. Wilson/rectangle gauge force, traceless antihermitian projection, and the
   matrix exponential gauge update.

The flow operator is `sum_mu D_mu(D_mu psi) = -sum_mu D_mu^dagger D_mu psi`.
There are no mixed-direction terms. Fat and Naik links are interleaved, with
one-half multiplying the forward-minus-backward stencil. Thin links stay
unphased. Smearing uses the same `newHisq(0,1)` coefficients and phase convention
as the CPU reference. Nonzero flow mass and gauge actions containing adjoint
plaquette or parallelogram terms are rejected.

| Source | Responsibility |
| --- | --- |
| `src/alphas/flowstepgpu.nim` | Persistent fields and the three coupled RK stages |
| `src/alphas/gaugeflowgpu.nim` | Gauge force/update and two-level HISQ smearing |
| `src/alphas/hisqflowgpu.nim` | Directional stencil and resident Laplacian |
| `src/comms/halodevice.nim` | Device halo packing, local copies and compact MPI messages |
| `src/alphas/hisqflowutils.nim` | Select CPU or accelerator flow; invoke existing measurements |

Halo exchanges copy local neighbors on device and transfer only packed remote
boundary values through host memory for QMP/MPI. This does not require GPU-aware
MPI. Exchanges and kernels are synchronous; communication overlap is future
work. Explicit cleanup releases workspace allocations. Captured sequences
share stable storage, and GPU metadata stays valid when its allocation table
grows during nested conversions.

The matrix exponential specializes QEX's existing order-four expm1 algorithm
with 20 scaling/squaring steps. Its numerical algorithm is unchanged; selecting
it explicitly avoids inlining unused runtime branches into the device kernel.

Inversion, measurement reductions and file I/O remain on CPU. At the end of
each complete flow step, the integrator downloads one snapshot for the existing
measurement interface. No full-field download is needed between RK stages or
between the two applications of a directional operator. The standalone
`directional`/`laplacian` convenience methods still accept host fields and
return host-visible workspace storage; `laplacianInto` is the resident API.

This is a correctness-oriented implementation. Explicit path products repeat
some matrix multiplications, and rectangular halos reserve unused edge/corner
storage. Several full gauge and per-source RK work fields are retained. Measure
memory use and throughput on the intended decomposition before production;
small-lattice correctness checks establish neither memory fit nor speedup.

## Build

Requires Nim 2, a SYCL-capable C++ compiler, compatible QMP/QIO libraries,
and their MPI runtime. The copied Stampede3 configuration can be used with:

```bash
source ./load-mpi-env.sh
./scripts/build-hisq-sycl.sh SYCL check_sycl_smoke
./scripts/build-hisq-sycl.sh SYCL check_hisq_gpu
./scripts/build-hisq-sycl.sh SYCL check_gaugeflow_gpu
./scripts/build-hisq-sycl.sh SYCL check_device_halo
./scripts/build-hisq-sycl.sh SYCL check_gpu_mem
./scripts/build-hisq-sycl.sh SYCL hisqflow
```

Products and separate per-target caches go under `build-sycl/SYCL/`.
The script defaults to `mpicxx` with `I_MPI_CXX=icpx`; override `SYCL_CXX`, `CXX`, `NIM`,
`QMP_DIR`, `QIO_DIR`, `VLEN`, or `BUILD_JOBS` for another installation.
Build concurrency defaults to two compiler processes. The source GPU memory,
field and sequence adapters include fixes required by Nim 2.0.17. SYCL
range-kernel indexing uses the oneAPI 2024 `experimental::this_item` binding.

The application's existing command-line and JSON inputs are unchanged.
Select a device using your SYCL runtime's device selector and verify the
printed `SYCL device:` line. QEX assigns rank modulo the number of visible
devices on the selected platform; arrange rank placement/device visibility
for the target machine.

## Production target: Perlmutter A100

The intended production system is **NERSC Perlmutter**, whose GPU nodes have
four NVIDIA A100s. Use `SYCL_TARGET=a100` (`sm_80`) there; keep
`SYCL_TARGET=h100` (`sm_90`) for interim Stampede3 testing. See
[Perlmutter build and validation](PERLMUTTER_SYCL.md) for the Cray MPI setup,
dependency rebuild and four-GPU Slurm job. Both profiles use the same physics
kernels. `QEX_BUILD_ROOT` can isolate all products, including the CPU reference,
for each machine/toolchain. Stampede3-linked binaries are not deployable
Perlmutter binaries.

## Interim target: Stampede3 NVIDIA H100

The interim GPU target is **NVIDIA H100** (compute capability 9.0,
`sm_90`, per [NVIDIA's CUDA GPU table](https://developer.nvidia.com/cuda/gpus)).
The H100 profile uses the **SYCL CUDA backend**. It requires a SYCL compiler
and runtime built with NVIDIA support, such as CUDA-enabled LLVM DPC++ or a
matching oneAPI installation with the NVIDIA plugin, plus a compatible CUDA
toolkit and driver. The existing Stampede3 compiler can generate the NVIDIA
kernels, but the inspected runtime installation does not contain its CUDA
adapter (`libpi_cuda.so`). Changing the device selector alone does not add
that adapter.

On the H100 machine, load its compiler, CUDA and MPI environment. Use QMP/QIO
built for that machine's MPI stack; the copied Stampede3 dependencies are only
verified with the existing Intel MPI environment. Then configure:

```bash
export SYCL_TARGET=h100
export SYCL_CXX=/path/to/cuda-enabled/compiler/bin/icpx  # or clang++
export CUDA_PATH=/path/to/cuda
export QMP_DIR=/path/to/target-mpi/qmp
export QIO_DIR=/path/to/target-mpi/qio
./scripts/build-hisq-sycl.sh SYCL hisqflow
```

`CXX` defaults to `mpicxx`. The build script sets the underlying compiler
through `I_MPI_CXX`, `OMPI_CXX`, or `MPICH_CXX` as appropriate; explicit values
already set by the caller take precedence. H100 output and caches are isolated
under `build-sycl/SYCL-h100/`.

Both compilation and linking receive:

```text
-fsycl-targets=nvptx64-nvidia-cuda
-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_90
--cuda-path=<CUDA_PATH>
```

These flags follow the [LLVM SYCL CUDA build and architecture instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).
QEX supplies `-fsycl`. On an allocated H100 node, run the validation suite:

```bash
SYCL_TARGET=h100 ./scripts/check-hisq-sycl.sh
```

The suite forces `ONEAPI_DEVICE_SELECTOR=cuda:gpu`, checks that all visible
GPUs are NVIDIA H100s with FP64 and device USM, and stops on failure. Its
preflight is compiled as a host-only SYCL program; the following QEX smoke
and physics programs exercise the actual `sm_90` kernels. Logs go to
`build-sycl/validation-h100/`. Use the same CUDA selector when running the
application with its existing JSON/command-line arguments:

```bash
ONEAPI_DEVICE_SELECTOR=cuda:gpu ./build-sycl/SYCL-h100/hisqflow ...
```

The suite uses two MPI ranks. With two visible H100s on one node, QEX's
rank-modulo-device mapping gives each rank a different GPU. With one visible
GPU both ranks share it, which does not validate communication between GPUs.
The validation script's shared-memory MPI defaults apply only to one node;
production multi-node launching must follow the target cluster's setup.

### Stampede3 H100 build and launch status at checkpoint `c949ada`

The H100 smoke test, physics test and full `hisqflow` application have all
compiled and linked successfully for `sm_90`, using the existing Intel oneAPI
2024.0 / Intel MPI environment and CUDA 12.4 from the NVIDIA HPC SDK:

```bash
source ./load-mpi-env.sh
export CUDA_PATH=/home1/apps/nvidia/Linux_x86_64/24.5/cuda
export SYCL_TARGET=h100
./scripts/build-hisq-sycl.sh SYCL check_sycl_smoke
./scripts/build-hisq-sycl.sh SYCL check_hisq_gpu
./scripts/build-hisq-sycl.sh SYCL check_gaugeflow_gpu
./scripts/build-hisq-sycl.sh SYCL check_device_halo
./scripts/build-hisq-sycl.sh SYCL check_gpu_mem
./scripts/build-hisq-sycl.sh SYCL hisqflow
```

The CUDA path is used directly; loading an NVIDIA compiler module is not
needed and could change the MPI/compiler environment. Build logs are under
`build-sycl/h100-checks/`. H100 binaries are in `build-sycl/SYCL-h100/`.
The generic smoke test still passes, and the H100 preflight correctly rejects
CPU-only execution. No H100 kernel has been executed yet.

Stampede3 has an `h100` partition; the current scheduler listing and
[TACC documentation](https://docs.tacc.utexas.edu/hpc/stampede3/#gpu-nodes)
identify that as its H100 queue. No partition named experimental appeared. A hidden `systest-h100` partition
is restricted to group `G-25072`; this account belongs to `G-81591` and is not
in that allowed group. A Slurm `--test-only` check accepted the
partition/account request but, on November 5, estimated a start on November 9;
this was an estimate, not a reservation. No GPU job was submitted. The login
node has no H100, and the inspected Intel compiler installations (2024.0,
2025.1 and 2026.0) lack a CUDA runtime adapter. A compatible CUDA-enabled SYCL
runtime and a GPU allocation are still required. Runtime availability on
compute nodes has not been checked.

Once the runtime is available, submit the prepared single-node validation
job **from this repository's root**:

```bash
sbatch scripts/validate-hisq-h100-stampede3.sbatch
```

It uses the existing `TG-MCA93S002` account and the node-exclusive `h100`
partition, which advertises no GPU GRES. Override the account with `sbatch -A`
if needed. `H100_SYCL_ENV=/absolute/path/to/setup.sh` can supply additional
site runtime setup after `load-mpi-env.sh`; it must preserve compatibility
with the prebuilt oneAPI 2024 binaries and Intel MPI dependencies. The job
checks GPU visibility, then runs the prebuilt numerical programs with
`SKIP_BUILD=1`. The small host-only device preflight is still compiled.
It stops if the runtime cannot expose an H100. Do not submit it with the
current CPU-only runtime expecting GPU validation to succeed.

For a different SYCL/MPI toolchain, rebuild the binaries and dependencies
before submitting. The earlier numerical results below are SYCL CPU results.

## Validation

`tests/backend/check_hisq_gpu.nim` includes the actual CPU implementation as
its reference. It checks:

- free-field HISQ eigenvalues, including antiperiodic temporal boundaries;
- all four directional operators against CPU fat/Naik operators;
- the directional Laplacian on random gauge fields;
- negative semidefiniteness and exclusion of mixed-direction products;
- unchanged thin gauge fields and rejection of nonzero mass;
- source/link updates using the same workspace;
- all three color propagators and gauge fields after two coupled RK steps.

`check_gaugeflow_gpu.nim` independently compares Wilson and rectangle forces,
all eight fat/Naik links, and each of the three gauge RK updates with the CPU
reference. `check_device_halo.nim` compares local and remote boundary values
with QEX's host gather. `check_gpu_mem.nim` checks metadata pointer stability
across table growth. `check_hisq_gpu -rectangle:1` repeats the coupled test with
a rectangle action. Floating-point checks explicitly reject nonfinite results.

Comparisons use FP64 and relative tolerance `1e-11`. A source-level CPU
fallback check is available with `./scripts/build-hisq-sycl.sh CPU check_hisq_gpu`.

For a small, single-host validation on the available Stampede3 SYCL CPU device:

```bash
I_MPI_FABRICS=shm OMP_NUM_THREADS=2 CL_CONFIG_CPU_MAX_COMPUTE_UNITS=2 \
  ONEAPI_DEVICE_SELECTOR=opencl:cpu ./build-sycl/SYCL/check_hisq_gpu
```

`I_MPI_FABRICS=shm` is for these local checks, not for a multi-node job.
GPU hardware validation and GPU performance measurements remain required.

Run the reproducible single-host suite with `./scripts/check-hisq-sycl.sh`.
Its default selector is `opencl:cpu`; set `ONEAPI_DEVICE_SELECTOR` to the target
GPU selector to validate on GPU hardware. Build/run logs are written to
`build-sycl/validation/` and any failed command stops the suite.

The two-rank CPU-reference comparison uses an 8⁴ lattice because the existing
CPU `ShiftB` implementation rejects length-three shifts when an outer local
extent is below three. An independent analytic test (`-haloOnly:1`) bypasses
that reference and checks Naik halos on a 2×4×4×8 lattice with two ranks. This
also covers a hop longer than a global lattice extent. The halo mapper now
normalizes coordinates modulo the full extent, including repeated wraps.

### Resident gauge and fermion flow validation, November 5, 2026

The complete `scripts/check-hisq-sycl.sh` suite passed with Nim 2.0.17,
Intel oneAPI 2024.0, Intel MPI 2021.11, VLEN=8 and FP64. SYCL ran on the
Intel OpenCL CPU device (Xeon Platinum 8468), with two host threads and two
OpenCL compute units. These results validate device allocations and kernels,
but do not establish NVIDIA hardware correctness or performance.

| Check | Result |
| --- | --- |
| Optimized CPU fallback, two coupled RK steps | Passed; fermion error at most 3.57e-15 |
| SYCL Wilson flow, one and two MPI ranks | Passed; coupled fermion error at most 2.35e-15; gauge error below 2.74e-16 |
| SYCL rectangle flow, two coupled steps | Passed; fermion error below 7.87e-16 |
| Wilson/rectangle forces, one and two ranks | Passed; error below 3.09e-16 |
| All eight fat/Naik links, one and two ranks | Passed; error below 3.98e-13 |
| Each gauge RK stage, both actions, one and two ranks | Passed; error below 1.94e-16 |
| Packed halo exchange, narrow two-rank domains | Passed against host gather |
| Analytic repeated-wrap Naik halo | Passed; error 2.90e-16 |
| Imported device math and metadata-table growth | Passed |
| Workspace cleanup | No GPU allocations or metadata entries remain after cleanup |
| Generic SYCL full application | Compiled and linked |
| A100 sm_80 math smoke test and full application | Cross-compiled and linked on Stampede3; not executed on A100 |
| Original CPU-only application | Nim C generation passed without `hisqFlowGpu` |

Numerical logs are under `build-sycl/validation/`. Application build logs are
`build-sycl/resident-generic-app-build.log`, `resident-a100-app-build.log`,
and `resident-original-cpu-app.log`. The CPU fallback additionally passed with
one host thread and with the rectangle action. Scalar matrix helpers are
expanded inside CPU kernels as they are for device kernels; the optimized
CPU checks cover this path as well as the original CPU reference.

Backend fixes preserve imported math calls during Nim inlining, route the
projection's FP64 functions through SYCL's device math for NVIDIA, and keep
allocation metadata and captured sequence storage stable. Rebuild all
validation binaries before using the H100 or Perlmutter job scripts. The
older NVIDIA binaries from the checkpoint below cover only the earlier port.

No H100/A100 kernel execution, Perlmutter MPI integration, production lattice
memory validation, or GPU timing has been performed. The full application's
inversion/configuration-I/O/measurement workflow has not been run end to end.

### Earlier directional-only validation at checkpoint `c949ada`

These results predate the resident gauge/smearing/RK implementation above.
Recorded November 5, 2026.

Environment: Nim 2.0.17, Intel oneAPI 2024.0, Intel MPI 2021.11, VLEN=8,
FP64, two OpenMP threads per rank. SYCL selected Intel OpenCL CPU devices
(`Intel(R) Xeon(R) Platinum 8468`); no GPU was available on this host.

| Check | Result |
| --- | --- |
| CPU fallback directional operator, cleanup and two RK steps | Passed; maximum relative error 3.04e-14 |
| SYCL kernel allocation, execution and copy-back smoke test | Passed |
| SYCL directional operator and two RK steps, 4×4×4×8, one rank | Passed; maximum relative error 3.15e-14 |
| SYCL directional operator and two RK steps, 8⁴, two ranks | Passed; maximum relative error 5.88e-14 |
| Analytic Naik halos, 2×4×4×8, two ranks | Passed; relative error 2.90e-16 |
| Full `hisqflow` application with generic SYCL offload | Compiled and linked |
| H100 `sm_90` smoke, physics and full application binaries | Compiled and linked with CUDA 12.4; not executed on H100 |
| A100 `sm_80` smoke, physics and full application binaries | Compiled and linked on Stampede3 with CUDA 12.4; not Perlmutter-linked or executed on A100 |
| CPU fallback, four ranks, 8×8×8×16, operator and two RK steps | Passed; maximum relative error 6.09e-15 |
| Original CPU-only application without `hisqFlowGpu` | Nim C generation passed |

The propagated fields after two coupled steps differed from the CPU reference
by at most 3.59e-17 relative; the evolved gauge fields matched exactly. These
are numerical validation results, not representative performance timings.
The validation runs the actual coupled integrator, but the full application's
configuration-file/inversion/measurement workflow has not been run end to end
with this new binary. Production GPU runs require a suitable GPU allocation
and further device-specific validation.

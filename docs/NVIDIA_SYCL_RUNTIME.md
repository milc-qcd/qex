# CUDA-enabled SYCL runtime on Stampede3

A user-local LLVM/DPC++ **v7.2.0** compiler and runtime is installed at:

```text
/work2/11599/mdai07/stampede3/opt/llvm-sycl-v7.2.0-cuda
```

This supplies `libur_adapter_cuda.so`, the CUDA adapter missing from the site's
oneAPI installations. The compiler is Clang 23.1.1, upstream commit
`5680988bb38bd45f13081394fba43588c5f5e395`. The complete compiler, headers and
runtime come from the same [official tagged release](https://github.com/intel/llvm/releases/tag/v7.2.0).
The downloaded Linux archive was checked against its published SHA-256:

```text
385586a25e75ac727883f0be5e442dec0ee81b3c20f9a29037a79921ad2f1728
```

The installer records the source and checksum in `qex-install-manifest.json`.
The original site compilers, copied CPU repository, and previously validated
Intel-GPU binaries are preserved. Installing this adapter does not establish
NVIDIA hardware correctness; the H100 job below must pass on actual hardware.

## Setup and reproducible installation

From the `qex-gpu-sycl` repository:

```bash
python3 scripts/install-nvidia-sycl.py
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
```

The installer uses a pinned, verified upstream binary archive. Override its
`--prefix` and `--cache` arguments if needed, then set
`QEX_NVIDIA_SYCL_PREFIX` before sourcing the environment helper. Existing
unrelated installation directories are not overwritten.

The helper selects the following components:

| Component | Setting |
| --- | --- |
| SYCL compiler | `$QEX_NVIDIA_SYCL_PREFIX/bin/clang++` |
| CUDA adapter and SYCL runtime | `$QEX_NVIDIA_SYCL_PREFIX/lib`, first in library search paths |
| CUDA toolkit | `/home1/apps/nvidia/Linux_x86_64/24.5/cuda` (CUDA 12.4) |
| MPI | Stampede3 Intel MPI 2021.11 |
| Rebuilt QMP/QIO | `deps-sycl/llvm-v7.2.0/{qmp,qio}` |
| Build products | `build-sycl/llvm-v7.2.0/` |
| H100 / A100 architecture | `nvidia_gpu_sm_90` / `nvidia_gpu_sm_80` |
| CPU OpenMP | Intel `libiomp5`, with host-only compiler flags |

The environment helper removes old oneAPI compiler include directories from
`CPATH`, `C_INCLUDE_PATH` and `CPLUS_INCLUDE_PATH`; other dependency paths are
preserved. It sets both the MPI compiler wrappers and runtime library search
paths to the new toolchain. This avoids mixing old SYCL headers with the new
runtime. `CUDA_PATH`, dependency prefixes and `QEX_BUILD_ROOT` can be overridden
before sourcing it.

The new driver interprets ordinary `-fopenmp` together with a SYCL CUDA target
as also requesting OpenMP GPU offload. QEX's CPU threading instead uses:

```text
-Xarch_host -fopenmp=libiomp5
-idirafter /opt/intel/oneapi/compiler/2024.0/opt/compiler/include
```

These are supplied through QEX's existing `OMPFLAG` setting. The header path
supplies `omp.h` after the new compiler's headers. An actual two-thread OpenMP
region compiled and ran successfully with these flags and an H100 SYCL target.
`QEX_HOST_OMP_FLAGS` can override this Stampede3-specific host runtime setup.

The copied QMP archive contained Intel-specific `_intel_fast_memcpy` references.
QMP 2.5.4 and QIO 3.0.0 were rebuilt with LLVM's C compiler through Intel MPI's
`mpicc` wrapper, into separate prefixes. Reproduce that step with:

```bash
./scripts/build-flow-mpi-deps.sh
```

## Build and validation jobs

Submit the complete build matrix from the repository root:

```bash
sbatch scripts/build-nvidia-sycl-stampede3.sbatch
```

It builds and runs the CPU reference and generic SYCL CPU-device smoke/two-rank
checks, then builds the full application and all NVIDIA numerical tests for
H100 and A100. Build output goes to `build-sycl/llvm-v7.2.0/build-logs/`.
The CPU-only runs are explicitly separate from CUDA hardware validation.

H100 job submission uses the same environment and prebuilt binaries:

```bash
sbatch scripts/validate-hisq-h100-stampede3.sbatch
```

The job runs `nvidia-smi`, `ONEAPI_DEVICE_SELECTOR=cuda:gpu sycl-ls`, a strict
H100/CUDA/FP64/device-USM preflight, and the QEX numerical suite. There is no
CPU fallback in the device preflight. Results, binary checksums and an exit
status are written under:

```text
build-sycl/llvm-v7.2.0/validation-h100/job_JOBID/
```

An `exit-status.txt` value of `0` means the complete job passed. Review the
`device.log`, `smoke.log`, `sycl.log`, `mpi2.log`, `rectangle.log`,
`check_gaugeflow_gpu.log`, `gauge-mpi2.log`, `device-halo-mpi2.log`,
`narrow.log` and metadata checks as well. Nonzero status means validation
failed; a queued job or device enumeration alone is not a physics pass.

## Current run record

Build job **3686548** completed successfully on November 5, 2026, node
`c552-005`, elapsed **9 min 40 s**, exit status **0:0**. Results:

| Check | Result |
| --- | --- |
| CPU reference / two coupled steps | Passed |
| SYCL CPU-device smoke, transfers and imported math | Passed |
| Two-rank SYCL CPU-device coupled flow | Passed; maximum fermion relative error `5.72439e-15` |
| H100 device preflight, smoke, flow, gauge, halo and metadata programs | Built and linked |
| Full H100 `hisqflow` | Built and linked; host libraries resolve without missing dependencies |
| A100 device preflight, smoke, flow, gauge, halo and metadata programs | Built and linked |
| Full A100 `hisqflow` | Built and linked for Stampede3; not Perlmutter-linked |
| NVIDIA kernel execution | **Pending** |

H100 validation job **3686550** was submitted with `afterok:3686548`. Its build
dependency is now satisfied, and it is waiting for queue priority. No actual
start time was assigned at the final check. The initial `sbatch --test-only`
estimate was November 9; it was an estimate, not a reservation.
The job will execute the CUDA preflight and numerical suite when scheduled.

The [tracked installation/build record](validation/nvidia-runtime-20261105.json)
contains the release checksum and all H100/A100 binary hashes. Detailed build,
CPU-check, and dependency logs remain under `build-sycl/llvm-v7.2.0/`.
The login node lacks NVIDIA driver libraries (`libcuda.so.1` and
`libnvidia-ml.so.1`), as expected for this non-NVIDIA host. Actual adapter loading
and kernel execution must be checked on the H100 node; installing the adapter
and cross-compiling are not counted as a CUDA hardware pass.

## Perlmutter

A100 binaries built here are Stampede3-linked cross-builds. Perlmutter still
requires its own compatible CUDA-enabled toolchain environment, host OpenMP
runtime, QMP/QIO rebuilt against Cray MPICH, and actual A100 checks. Do not copy
Stampede3 Intel MPI libraries into a Perlmutter run. See
[PERLMUTTER_SYCL.md](PERLMUTTER_SYCL.md). Access to Perlmutter is deferred.

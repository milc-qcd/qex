# Perlmutter directional HISQ flow

Perlmutter is the production target; Stampede3 H100 nodes are an interim
validation option. [NERSC's architecture documentation](https://docs.nersc.gov/systems/perlmutter/architecture/)
lists four NVIDIA A100 GPUs per Perlmutter GPU node, with 40 GB or 80 GB per
GPU depending on the node type. Build the kernels for **`sm_80`**, using
`SYCL_TARGET=a100`. The H100 profile remains `sm_90`.

The resident flow implementation is described in [HISQ_SYCL.md](HISQ_SYCL.md).
Gauge force/update, HISQ smearing, directional fermion flow, RK combinations
and halo packing run on the selected SYCL device. MPI uses compact host-staged
messages. Inversion, measurements and I/O remain on CPU; a snapshot is downloaded
after each complete flow step. The workspace and rectangular halos retain
substantial storage; validate memory fit and throughput on the intended lattice.
The full 0.15 fm configuration 1000 passed on Stampede3 Intel Max 1550 GPUs;
see [the comparison record](CONFIG1000_VALIDATION.md). That result does not
establish A100 execution or Perlmutter MPI compatibility.

## Compiler and runtime

A CUDA-enabled LLVM/DPC++ v7.2.0 installation has been prepared on Stampede3;
[its setup record](NVIDIA_SYCL_RUNTIME.md) documents the compiler/runtime pairing
and build compatibility fixes. Its Intel MPI and host OpenMP settings are
Stampede3-specific; activate and validate the corresponding Perlmutter setup.

Use a CUDA-enabled LLVM DPC++/oneAPI compiler **and its matching CUDA runtime
adapter**, Nim 2, the site's CUDA toolkit, and Cray MPICH. The generic NERSC
C++ page shows an old `intel/2024.1.0` example, but the [February 19, 2026
system update](https://docs.nersc.gov/systems/perlmutter/timeline/#february-19-2026)
explicitly removed that module because its SYCL plugin was no longer
maintained. Do not rely on that module still being available. A generic LLVM
or NVIDIA HPC compiler alone does not establish SYCL CUDA support.

Activate a user/site installation of CUDA-enabled DPC++ and its matching
runtime, following its setup instructions. The [upstream CUDA backend build
instructions](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md)
cover building that toolchain if a suitable installation is unavailable.
The active module/runtime versions must be recorded and tested on Perlmutter;
this workspace has not connected to Perlmutter or inspected its modules.

Load the site's GNU programming environment, Cray MPICH and CUDA toolkit,
then use the actual installed paths:

```bash
# Run on Perlmutter, in the copied qex-gpu-sycl repository.
# Load/activate the site programming environment and CUDA-enabled SYCL runtime first.
export SYCL_CXX=/path/to/cuda-enabled-sycl/bin/icpx  # or that installation's clang++
export CUDA_PATH=/path/to/the/loaded/cuda/toolkit
export PATH=/path/to/nim/bin:$PATH
source scripts/load-perlmutter-env.sh
```

This helper selects A100 and puts all binaries/caches under
`build-sycl/perlmutter/`. It chooses `deps-perlmutter/qmp` and
`deps-perlmutter/qio`, avoiding the copied Intel MPI dependencies. It sets
`MPICH_GPU_SUPPORT_ENABLED=0` because the current MPI calls exchange host
buffers. This does **not** disable the SYCL device kernels. Device-buffer
MPI will require separate implementation and validation.

`CXX` points to `scripts/sycl-cray-cxx.py`. That adapter invokes the selected
SYCL compiler with include and library options obtained from the loaded Cray
`CC` wrapper using `--cray-print-opts=cflags` and `--cray-print-opts=libs`.
Library options follow the QMP/QIO libraries on the link command. This uses
[Cray's documented flag export interface](https://cpe.ext.hpe.com/docs/24.07/craype/cc.html),
without assuming that `CC` itself selects a SYCL compiler. It still requires
an MPI-compatible host C++ ABI and testing against the actual site toolchain.

## Rebuild MPI dependencies and application

Rebuild QMP/QIO against Perlmutter's Cray MPICH. Do not reuse the installed
Stampede3 libraries. Copy the existing source archives along with the code:
`deps-mpi/qmp-2.5.4.tar.gz` and `deps-mpi/qio-3.0.0.tar.gz` (or set
`QMP_TARBALL` and `QIO_TARBALL` to their locations). These source archives
are currently untracked files in the copied working tree; a plain Git clone
will not include those archives. The flow implementation is committed.

```bash
# Same environment as above; cc must be the Cray MPI C wrapper.
MPI_CC=cc ./scripts/build-flow-mpi-deps.sh
./scripts/build-hisq-sycl.sh CPU check_hisq_gpu
./scripts/build-hisq-sycl.sh SYCL check_cuda_device
./scripts/build-hisq-sycl.sh SYCL check_sycl_smoke
./scripts/build-hisq-sycl.sh SYCL check_hisq_gpu
./scripts/build-hisq-sycl.sh SYCL check_gaugeflow_gpu
./scripts/build-hisq-sycl.sh SYCL check_device_halo
./scripts/build-hisq-sycl.sh SYCL check_gpu_mem
./scripts/build-hisq-sycl.sh SYCL hisqflow
```

The dependency script extracts clean source trees under `deps-perlmutter/`
and preserves its sources and configure logs there. Override `QMP_DIR` and
`QIO_DIR` before sourcing the environment helper to use existing compatible
installations. CUDA compilation and linking both receive `sm_80`. All host
objects and dependencies must also be built on Perlmutter: successful A100
cross-compilation on Stampede3 does not establish MPI/CPU portability.

## Validate on an allocation

Submit from the repository root after building there, with the same module
and runtime environment:

```bash
sbatch -A YOUR_NERSC_GPU_ACCOUNT scripts/validate-hisq-perlmutter.sbatch
```

The script requests one GPU node, four GPUs, four MPI ranks and 15 minutes
in debug QoS, following [NERSC's Slurm job conventions](https://docs.nersc.gov/jobs/examples/).
It uses `srun --gpus-per-task=1 --gpu-bind=single:1`, giving each rank one
visible GPU. QEX then selects that visible GPU. Set `PERLMUTTER_SYCL_ENV`
to an absolute setup-script path if the job needs to restore the exact build
environment; the script sources it before the repository environment helper.
No project account is hardcoded.

The job checks A100 identity, CUDA backend, FP64 and device USM on all four
ranks, followed by a CPU reference run, device smoke tests, one- and four-rank
operator/RK comparisons, and a two-rank narrow-halo analytic test. The
four-rank reference uses 8×8×8×16 to avoid the CPU implementation's restrictions
on length-three shifts. Logs go to `build-sycl/perlmutter/validation-JOBID/`.
The job uses prebuilt programs and stops at the first failure. The device
preflight only checks capabilities; the subsequent tests execute the kernels.

This is a single-node validation job. Production multi-node decomposition,
MPI scaling, memory use, full application I/O and timings remain to be tested.
The full application retains its existing JSON and command-line interface.

## Resident-flow build status

The full application, including gauge force/update, HISQ smearing and fermion
RK kernels, cross-compiled and linked for `sm_80` on Stampede3. The device-math
smoke test also builds for `sm_80`. The one- and two-rank numerical suite passed
on the SYCL OpenCL CPU device; see [the recorded results](HISQ_SYCL.md#resident-gauge-and-fermion-flow-validation-november-5-2026).
This does not replace rebuilding against Perlmutter's MPI and validating on
its A100 devices. No Perlmutter access or job submission has been attempted.

## Earlier validation at checkpoint `c949ada`

The results below predate resident gauge/smearing/RK offload. See
[HISQ_SYCL.md](HISQ_SYCL.md) for the updated flow validation.

- The A100 (`sm_80`) smoke test, physics test and full `hisqflow` application
  compiled and linked successfully on Stampede3 using Intel oneAPI 2024.0,
  Intel MPI and CUDA 12.4. The embedded GPU target was verified in each binary.
  These are **not Perlmutter-linked binaries** and have not run on an A100.
  Artifacts are under `build-sycl/SYCL-a100/`; build logs are under
  `build-sycl/a100-checks/`.
- The CPU accelerator fallback passed the four-rank 8×8×8×16 operator and two
  coupled RK steps on Stampede3; maximum relative error was 6.09e-15, below
  the 1e-11 tolerance. The evolved gauge fields matched exactly.
- The dependency builder configured, built and installed clean QMP 2.5.4 and
  QIO 3.0.0 copies using local Intel MPI. The Cray MPICH build remains untested.
- The Cray adapter's argument quoting, compile/link separation, library order,
  and subprocess error handling were checked with mock compiler wrappers.
- The generic device preflight rejects CPU-only execution for both A100 and H100.
- No Perlmutter compiler/MPI integration, A100 execution or GPU timing has been
  performed. A Perlmutter login/allocation and a CUDA-enabled SYCL installation
  are needed for those checks. No Perlmutter job has been submitted.

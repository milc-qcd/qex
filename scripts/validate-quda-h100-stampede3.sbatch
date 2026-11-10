#!/usr/bin/env bash
#SBATCH --job-name=qex-quda-h100
#SBATCH --partition=h100
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=build-sycl/quda-h100-validation-%j.log
# Require a successful build and prepared configuration-1000 inputs. One rank
# per H100; run the small solver/flow tests before either full comparison.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
REPO=$PWD
source ./load-mpi-env.sh
source ./scripts/load-nvidia-sycl-env.sh
source ./scripts/load-quda-env.sh
INPUT=${QEX_QUDA_VALIDATION_ROOT:?Set the prepared validation directory}
BUILD_JOB=${QEX_QUDA_BUILD_JOB:?Set the QEX QUDA build job ID}
ATTEMPT="$INPUT/attempts/job_${SLURM_JOB_ID}"
mkdir -p "$INPUT/attempts"
mkdir "$ATTEMPT"
MONITOR=""
stop_monitor() {
  if [[ -n "$MONITOR" ]]; then
    kill "$MONITOR" 2>/dev/null || true
    wait "$MONITOR" 2>/dev/null || true
    MONITOR=""
  fi
}
finish() {
  local status=$?
  trap - EXIT
  stop_monitor
  printf '%s\n' "$status" > "$ATTEMPT/exit-status.txt"
  date -Is > "$ATTEMPT/finished.txt"
  exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
sha256sum -c "$QEX_BUILD_ROOT/quda-build-records/$BUILD_JOB/binaries.sha256" > "$ATTEMPT/binary-integrity.log"
(cd "$INPUT" && sha256sum -c SHA256SUMS.inputs) > "$ATTEMPT/input-integrity.log"
export ONEAPI_DEVICE_SELECTOR=cuda:gpu OMP_NUM_THREADS=2
export I_MPI_FABRICS=shm I_MPI_HYDRA_BOOTSTRAP=fork I_MPI_PIN=1 I_MPI_PIN_DOMAIN=24
export OMP_PROC_BIND=spread OMP_PLACES=cores SLURM_CPU_BIND=cores
export QUDA_RESOURCE_PATH="$ATTEMPT/tuning"
mkdir -p "$QUDA_RESOURCE_PATH" "$ATTEMPT/bin"
cp -p "$QEX_BUILD_ROOT/SYCL-h100-quda/"{hisqflow,check_quda_hisq} "$ATTEMPT/bin/"
cp -p "$INPUT/compare-flow-output.py" "$ATTEMPT/"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$ATTEMPT/nvidia-smi.txt"
sycl-ls > "$ATTEMPT/devices.txt"
"$QEX_BUILD_ROOT/SYCL-h100/check_cuda_device" H100 > "$ATTEMPT/device-preflight.log"
mpiexec -n 1 "$ATTEMPT/bin/check_quda_hisq" > "$ATTEMPT/single.log" 2>&1
mpiexec -n 2 "$ATTEMPT/bin/check_quda_hisq" > "$ATTEMPT/mpi2.log" 2>&1
grep -F 'PASS:' "$ATTEMPT/"{single,mpi2}.log
export OMP_NUM_THREADS=16
GAUGE_STEM=/scratch/11599/mdai07/lat015phys/l3248f211b580m002426m06730m8447a
for solver in qex quda; do
  OUT="$ATTEMPT/$solver"
  mkdir "$OUT"
  cp -p "$INPUT/"{flow-test.json,lattice-015phys.json,validate_output.py} "$OUT/"
  nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv --loop=1 \
    > "$OUT/gpu-telemetry.csv" 2> "$OUT/gpu-telemetry.err" &
  MONITOR=$!
  /usr/bin/time -v -o "$OUT/resource-usage.txt" \
    ibrun "$ATTEMPT/bin/hisqflow" --solver="$solver" \
      --flow-json="$OUT/flow-test.json" --lattice-json="$OUT/lattice-015phys.json" \
      --configuration=1000 --base-filename="$GAUGE_STEM" --output="$OUT" \
    > "$OUT/run.log" 2>&1
  stop_monitor
  grep -F 'SYCL device: NVIDIA H100' "$OUT/run.log"
  grep -F "HISQ solver backend: $solver" "$OUT/run.log"
  python3 "$OUT/validate_output.py" 015phys 1000 > "$OUT/completeness.log"
  python3 "$ATTEMPT/compare-flow-output.py" --reference "$INPUT/reference" --candidate "$OUT" \
    --configuration 1000 --rtol 1e-8 --atol 1e-10 \
    --report "$OUT/comparison.json" --csv "$OUT/comparison.csv" | tee "$OUT/comparison.log"
done
python3 "$ATTEMPT/compare-flow-output.py" --reference "$ATTEMPT/qex" --candidate "$ATTEMPT/quda" \
  --configuration 1000 --rtol 1e-8 --atol 1e-10 \
  --report "$ATTEMPT/solver-comparison.json" --csv "$ATTEMPT/solver-comparison.csv"
echo 'PASS: QUDA solver/flow tests and full configuration 1000 comparisons'

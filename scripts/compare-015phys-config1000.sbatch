#!/usr/bin/env bash
#SBATCH --job-name=015-sycl-cfg1000
#SBATCH --partition=pvc
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=3
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Submit from the prepared, isolated validation directory. Set
# QEX_DEVICE_SELECTOR=level_zero:gpu for Intel GPUs, or opencl:cpu for host
# device-kernel correctness checks. Record the selector in the run manifest.
set -euo pipefail
ROOT=${QEX_VALIDATION_ROOT:-${SLURM_SUBMIT_DIR:?}}
BASE=/scratch/11599/mdai07/lat015phys/l3248f211b580m002426m06730m8447a
CONFIG=1000
cd "$ROOT"
sha256sum -c SHA256SUMS.inputs
[[ "${SLURM_NTASKS:?}" == 32 && "${SLURM_JOB_NUM_NODES:?}" == 1 ]]
module purge
module load intel/24.0 impi/21.11
export ONEAPI_DEVICE_SELECTOR=${QEX_DEVICE_SELECTOR:-opencl:cpu}
export OMP_NUM_THREADS=2 OMP_PROC_BIND=spread OMP_PLACES=cores
export CL_CONFIG_CPU_MAX_COMPUTE_UNITS=2
export I_MPI_FABRICS=shm I_MPI_PIN=1 I_MPI_PIN_DOMAIN="${SLURM_CPUS_PER_TASK:?}"
export SLURM_CPU_BIND=cores
ATTEMPT="$ROOT/attempts/job_${SLURM_JOB_ID:?}"
mkdir -p "$ROOT/attempts"
mkdir "$ATTEMPT"
RUN_WORK=$(mktemp -d "/tmp/qex-sycl-${SLURM_JOB_ID}-${CONFIG}.XXXXXX")
preserve() {
  local status=$?
  trap - EXIT
  set +e
  for name in correlator_C0p0_1000 gaugeobs_C0p0_1000 validation_1000.log comparison.json comparison.csv comparison.log; do
    if [[ -f "$RUN_WORK/$name" ]]; then cp -p "$RUN_WORK/$name" "$ATTEMPT/$name"; fi
  done
  python3 - "$ATTEMPT/status.json" "$status" <<'PY'
import datetime, json, os, socket, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"job_id": os.environ["SLURM_JOB_ID"],
    "host": socket.gethostname(), "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "exit_code": int(sys.argv[2]), "backend": "SYCL", "device_selector": os.environ["ONEAPI_DEVICE_SELECTOR"]}, indent=2)+"\n")
PY
  exit "$status"
}
trap preserve EXIT
trap 'exit 143' TERM
cp -p bin/hisqflow flow-test.json lattice-015phys.json validate_output.py compare-flow-output.py "$RUN_WORK/"
cd "$RUN_WORK"
if command -v sycl-ls >/dev/null; then sycl-ls > "$ATTEMPT/device-inventory.txt"; fi
echo "Start: $(date -Is); node: ${SLURM_JOB_NODELIST}; configuration: $CONFIG"
echo "32 MPI ranks; 2 OpenMP threads/rank; ${SLURM_CPUS_PER_TASK} allocated cores/rank"
echo "SYCL selector: $ONEAPI_DEVICE_SELECTOR; CPU-device compute-unit limit: $CL_CONFIG_CPU_MAX_COMPUTE_UNITS"
echo "Reference: $ROOT/reference; attempt: $ATTEMPT; local work: $RUN_WORK"
/usr/bin/time -v -o "$ATTEMPT/resource-usage.txt" \
  ibrun ./hisqflow --flow-json="$RUN_WORK/flow-test.json" \
    --lattice-json="$RUN_WORK/lattice-015phys.json" --configuration="$CONFIG" \
    --base-filename="$BASE" --output="$RUN_WORK" 2>&1 \
  | tee "$RUN_WORK/run.log" > "$ATTEMPT/run.log"
python3 validate_output.py 015phys "$CONFIG" | tee validation_1000.log
python3 compare-flow-output.py --reference "$ROOT/reference" --candidate "$RUN_WORK" \
  --configuration "$CONFIG" --rtol 1e-8 --atol 1e-10 \
  --report comparison.json --csv comparison.csv | tee comparison.log
echo "Finished: $(date -Is)"

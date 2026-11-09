#!/usr/bin/env bash
#SBATCH --job-name=015-h100-cfg1000
#SBATCH --partition=h100
#SBATCH --account=TG-MCA93S002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Submit from an isolated directory with frozen inputs, binaries and environment
# scripts. The 32-rank layout and two host threads match the CPU/PVC comparison.
set -euo pipefail
ROOT=${QEX_VALIDATION_ROOT:-${SLURM_SUBMIT_DIR:?}}
REPO=${QEX_SYCL_ROOT:-/work2/11599/mdai07/stampede3/qex-gpu-sycl}
BASE=/scratch/11599/mdai07/lat015phys/l3248f211b580m002426m06730m8447a
CONFIG=1000
cd "$ROOT"
[[ "${SLURM_NTASKS:?}" == 32 && "${SLURM_JOB_NUM_NODES:?}" == 1 ]]
ATTEMPT="$ROOT/attempts/job_${SLURM_JOB_ID:?}"
mkdir -p "$ROOT/attempts"
mkdir "$ATTEMPT"
RUN_WORK=""
GPU_MONITOR=""
stop_monitor() {
  if [[ -n "$GPU_MONITOR" ]]; then
    kill "$GPU_MONITOR" 2>/dev/null || true
    wait "$GPU_MONITOR" 2>/dev/null || true
    GPU_MONITOR=""
  fi
}
preserve() {
  local status=$?
  trap - EXIT
  set +e
  stop_monitor
  if [[ -n "$RUN_WORK" ]]; then
    for name in correlator_C0p0_1000 gaugeobs_C0p0_1000 validation_1000.log comparison.json comparison.csv comparison.log; do
      if [[ -f "$RUN_WORK/$name" ]]; then cp -p "$RUN_WORK/$name" "$ATTEMPT/$name" || status=1; fi
    done
  fi
  python3 - "$ATTEMPT" "$status" <<'PY'
import csv, datetime, json, os, re, socket, sys
from pathlib import Path
root = Path(sys.argv[1])
status = {"job_id": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
          "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "exit_code": int(sys.argv[2]), "backend": "SYCL", "device_selector": "cuda:gpu"}
log = root / "run.log"
if log.exists():
    times = re.findall(r"Total time \(Init - Finalize\): ([\d.eE+-]+) seconds", log.read_text(errors="replace"))
    if times:
        status["application_elapsed_seconds"] = float(times[-1])
comparison = root / "comparison.json"
if comparison.exists():
    status["comparison_passed"] = json.loads(comparison.read_text())["passed"]
(root / "status.json").write_text(json.dumps(status, indent=2) + "\n")
telemetry = root / "gpu-telemetry.csv"
if telemetry.exists():
    devices = {}
    with telemetry.open() as stream:
        for raw in csv.DictReader(stream):
            row = {key.strip(): value.strip() for key, value in raw.items() if key and value}
            try:
                used = float(row["memory.used [MiB]"])
                uuid = row["uuid"]
                item = devices.setdefault(uuid, {"index": int(row["index"]), "name": row["name"],
                    "memory_total_mib": float(row["memory.total [MiB]"]), "peak_sampled_memory_used_mib": 0,
                    "samples": 0})
                item["peak_sampled_memory_used_mib"] = max(item["peak_sampled_memory_used_mib"], used)
                item["samples"] += 1
            except (KeyError, ValueError):
                continue
    (root / "gpu-memory-summary.json").write_text(json.dumps({
        "measurement": "nvidia-smi total device memory.used sampled every second during the application",
        "scope": "All processes/driver allocations on each GPU; short peaks between samples may be missed.",
        "devices": devices}, indent=2) + "\n")
PY
  exit "$status"
}
trap preserve EXIT
trap 'exit 143' TERM
sha256sum -c SHA256SUMS.inputs > "$ATTEMPT/input-integrity.log"
source "$ROOT/environment/load-mpi-env.sh"
export QEX_NVIDIA_SYCL_PREFIX="$REPO/../opt/llvm-sycl-v7.2.0-cuda"
export QEX_BUILD_ROOT="$REPO/build-sycl/llvm-v7.2.0"
export QEX_DEPS_ROOT="$REPO/deps-sycl/llvm-v7.2.0"
source "$ROOT/environment/load-nvidia-sycl-env.sh"
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
export OMP_NUM_THREADS=2 OMP_PROC_BIND=spread OMP_PLACES=cores
export I_MPI_FABRICS=shm I_MPI_PIN=1 I_MPI_PIN_DOMAIN="${SLURM_CPUS_PER_TASK:?}"
export SLURM_CPU_BIND=cores
hostname > "$ATTEMPT/host.txt"
module list > "$ATTEMPT/modules.txt" 2>&1
ldd "$ROOT/bin/hisqflow" > "$ATTEMPT/application-libraries.txt"
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv > "$ATTEMPT/nvidia-smi.txt"
sycl-ls > "$ATTEMPT/device-inventory.txt"
"$ROOT/bin/check_cuda_device" H100 | tee "$ATTEMPT/device-preflight.log"
RUN_WORK=$(mktemp -d "/tmp/qex-h100-${SLURM_JOB_ID}-${CONFIG}.XXXXXX")
cp -p bin/hisqflow flow-test.json lattice-015phys.json validate_output.py compare-flow-output.py "$RUN_WORK/"
cd "$RUN_WORK"
echo "Start: $(date -Is); node: ${SLURM_JOB_NODELIST}; configuration: $CONFIG"
echo "32 MPI ranks; 2 OpenMP threads/rank; ${SLURM_CPUS_PER_TASK} allocated cores/rank"
echo "SYCL selector: $ONEAPI_DEVICE_SELECTOR"
echo "Reference: $ROOT/reference; attempt: $ATTEMPT; local work: $RUN_WORK"
nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv --loop=1 \
  > "$ATTEMPT/gpu-telemetry.csv" 2> "$ATTEMPT/gpu-telemetry.err" &
GPU_MONITOR=$!
/usr/bin/time -v -o "$ATTEMPT/resource-usage.txt" \
  ibrun ./hisqflow --flow-json="$RUN_WORK/flow-test.json" \
    --lattice-json="$RUN_WORK/lattice-015phys.json" --configuration="$CONFIG" \
    --base-filename="$BASE" --output="$RUN_WORK" 2>&1 \
  | tee "$RUN_WORK/run.log" > "$ATTEMPT/run.log"
stop_monitor
grep -F 'SYCL device: NVIDIA H100' "$ATTEMPT/run.log"
python3 validate_output.py 015phys "$CONFIG" | tee validation_1000.log
python3 compare-flow-output.py --reference "$ROOT/reference" --candidate "$RUN_WORK" \
  --configuration "$CONFIG" --rtol 1e-8 --atol 1e-10 \
  --report comparison.json --csv comparison.csv | tee comparison.log
echo "Finished: $(date -Is)"

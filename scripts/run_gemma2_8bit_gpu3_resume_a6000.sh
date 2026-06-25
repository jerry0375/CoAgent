#!/usr/bin/env bash
set -u

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/gemma2_8bit_gpu3_resume_a6000_master.log"
PIDFILE="$LOG_DIR/gemma2_8bit_gpu3_resume_a6000.pid"
HEARTBEAT="$LOG_DIR/gemma2_8bit_gpu3_resume_a6000.heartbeat"
CONFIG="$ROOT/configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json"
WORK_DIR="$ROOT/outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000/general_base_gemma2_9b_it_coagent_conservative_train64"
WPO="$WORK_DIR/follower/general_base_gemma2_9b_it_coagent_conservative_train64_follower_wpo.jsonl"

mkdir -p "$LOG_DIR"
echo $$ > "$PIDFILE"
exec >> "$LOG" 2>&1

PYTHON="/opt/conda/bin/python"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CURRENT_STAGE="starting"
log() { echo "[$(date '+%F %T')] $*"; }

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      echo "model=gemma2_9b"
      echo "mode=8bit_gpu3_resume"
      echo "stage=${CURRENT_STAGE}"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits || true
    } > "$HEARTBEAT"
    sleep 60
  done
}

heartbeat_loop &
HB=$!
cleanup() { kill "$HB" >/dev/null 2>&1 || true; }
trap cleanup EXIT

wait_gpu3() {
  CURRENT_STAGE="waiting_gpu3"
  while true; do
    "$PYTHON" - <<'PY'
import subprocess
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free[int(idx)] = int(mem)
print(f"gpu3_free_mib={free.get(3, 0)}", flush=True)
raise SystemExit(0 if free.get(3, 0) >= 30000 else 1)
PY
    if [ "$?" -eq 0 ]; then
      log "GPU3 ready for 8bit DPO/eval"
      return 0
    fi
    log "GPU3 free memory below threshold; sleep 120s"
    sleep 120
  done
}

prepare_resume() {
  CURRENT_STAGE="prepare_resume"
  if [ ! -s "$WPO" ]; then
    log "missing follower WPO: $WPO"
    exit 2
  fi
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
cfg_path = Path("configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json")
cfg = json.loads(cfg_path.read_text())
cfg.setdefault("training", {})["load_in_8bit"] = True
cfg["training"]["train_max_length"] = 512
cfg.setdefault("model", {})["cuda_visible_devices"] = "3"
cfg["model"]["device"] = "cuda:0"
cfg.setdefault("parallel_sampling", {})["enabled"] = True
cfg["parallel_sampling"]["num_shards"] = 1
cfg["parallel_sampling"]["cuda_visible_devices"] = ["3"]
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
json.loads(cfg_path.read_text())
print(json.dumps({
    "config": str(cfg_path),
    "load_in_8bit": cfg["training"]["load_in_8bit"],
    "train_max_length": cfg["training"]["train_max_length"],
    "wpo_lines": sum(1 for _ in Path("outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000/general_base_gemma2_9b_it_coagent_conservative_train64/follower/general_base_gemma2_9b_it_coagent_conservative_train64_follower_wpo.jsonl").open()),
}, ensure_ascii=False))
PY
  for role in follower leader; do
    if [ -d "$WORK_DIR/adapters/$role" ] && [ ! -f "$WORK_DIR/adapters/$role/adapter_model.safetensors" ]; then
      log "remove incomplete adapter dir: $WORK_DIR/adapters/$role"
      rm -rf "$WORK_DIR/adapters/$role"
    fi
  done
}

write_summary() {
  local status="$1"
  CURRENT_STAGE="summary"
  "$PYTHON" - "$status" <<'PY'
import json, sys
from pathlib import Path
status = sys.argv[1]
work = Path("outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000/general_base_gemma2_9b_it_coagent_conservative_train64")
out = Path("outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000/gemma2_8bit_gpu3_resume_summary.json")
eval_summaries = sorted((work / "eval").glob("*summary.json"))
data = {
    "status": status,
    "config": "configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json",
    "load_in_8bit": True,
    "wpo_lines": sum(1 for _ in (work / "follower/general_base_gemma2_9b_it_coagent_conservative_train64_follower_wpo.jsonl").open()),
    "follower_adapter_exists": (work / "adapters/follower/adapter_model.safetensors").exists(),
    "leader_adapter_exists": (work / "adapters/leader/adapter_model.safetensors").exists(),
    "eval_summary_path": str(eval_summaries[-1]) if eval_summaries else None,
}
if eval_summaries:
    data["eval_summary"] = json.loads(eval_summaries[-1].read_text())
out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(data, ensure_ascii=False, indent=2))
PY
}

main() {
  log "============================================================"
  log "START gemma2 8bit GPU3 resume"
  prepare_resume
  wait_gpu3
  CURRENT_STAGE="strict_alternating_smoke"
  "$PYTHON" scripts/main.py strict-alternating-smoke --config "$CONFIG"
  rc=$?
  log "strict_alternating_smoke returncode=$rc"
  if [ "$rc" -eq 0 ]; then
    write_summary "completed"
  else
    write_summary "failed_rc_${rc}"
  fi
  exit "$rc"
}

main "$@"

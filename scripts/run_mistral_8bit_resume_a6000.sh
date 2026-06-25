#!/usr/bin/env bash
set -u

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/mistral_8bit_resume_a6000_master.log"
PIDFILE="$LOG_DIR/mistral_8bit_resume_a6000.pid"
HEARTBEAT="$LOG_DIR/mistral_8bit_resume_a6000.heartbeat"
CONFIG="$ROOT/configs/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_gpu2_sampling01.json"
WORK_DIR="$ROOT/outputs/mistral_7b_coagent_conservative_train64_a6000/general_base_mistral_7b_instruct_v03_coagent_conservative_train64"
WPO="$WORK_DIR/follower/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_follower_wpo.jsonl"

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
      echo "model=mistral_7b"
      echo "mode=8bit_resume"
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

wait_gpu2() {
  CURRENT_STAGE="waiting_gpu2"
  while true; do
    "$PYTHON" - <<'PY'
import subprocess
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free[int(idx)] = int(mem)
print(f"gpu2_free_mib={free.get(2, 0)}", flush=True)
raise SystemExit(0 if free.get(2, 0) >= 30000 else 1)
PY
    if [ "$?" -eq 0 ]; then
      log "GPU2 ready for 8bit DPO/eval"
      return 0
    fi
    log "GPU2 free memory below threshold; sleep 120s"
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
cfg_path = Path("configs/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_gpu2_sampling01.json")
cfg = json.loads(cfg_path.read_text())
cfg.setdefault("training", {})["load_in_8bit"] = True
cfg["training"]["train_max_length"] = 1024
cfg.setdefault("model", {})["cuda_visible_devices"] = "2"
cfg["model"]["device"] = "cuda:0"
cfg.setdefault("parallel_sampling", {})["enabled"] = True
cfg["parallel_sampling"]["num_shards"] = 2
cfg["parallel_sampling"]["cuda_visible_devices"] = ["0", "1"]
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
json.loads(cfg_path.read_text())
print(json.dumps({
    "config": str(cfg_path),
    "load_in_8bit": cfg["training"]["load_in_8bit"],
    "train_max_length": cfg["training"]["train_max_length"],
    "wpo_lines": sum(1 for _ in Path("outputs/mistral_7b_coagent_conservative_train64_a6000/general_base_mistral_7b_instruct_v03_coagent_conservative_train64/follower/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_follower_wpo.jsonl").open()),
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
work = Path("outputs/mistral_7b_coagent_conservative_train64_a6000/general_base_mistral_7b_instruct_v03_coagent_conservative_train64")
out = Path("outputs/mistral_7b_coagent_conservative_train64_a6000/mistral_8bit_resume_summary.json")
eval_summaries = sorted((work / "eval").glob("*summary.json"))
data = {
    "status": status,
    "config": "configs/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_gpu2_sampling01.json",
    "load_in_8bit": True,
    "wpo_lines": sum(1 for _ in (work / "follower/general_base_mistral_7b_instruct_v03_coagent_conservative_train64_follower_wpo.jsonl").open()),
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
  log "START mistral 8bit resume"
  prepare_resume
  wait_gpu2
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

#!/usr/bin/env bash
set -u

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/gemma2_gpu3_only_a6000_master.log"
PIDFILE="$LOG_DIR/gemma2_gpu3_only_a6000.pid"
HEARTBEAT="$LOG_DIR/gemma2_gpu3_only_a6000.heartbeat"
STATEFILE="$LOG_DIR/gemma2_gpu3_only_a6000.state"
CONFIG="$ROOT/configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json"
MODEL_DIR="/workspace/models/gemma-2-9b-it"
SUMMARY_DIR="$ROOT/outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000"
WORK_DIR="$SUMMARY_DIR/general_base_gemma2_9b_it_coagent_conservative_train64"

mkdir -p "$LOG_DIR" "$SUMMARY_DIR"
echo $$ > "$PIDFILE"
exec >> "$LOG" 2>&1

PYTHON="/opt/conda/bin/python"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CURRENT_STAGE="starting"
write_state() { echo "stage=$CURRENT_STAGE" > "$STATEFILE"; }
log() { echo "[$(date '+%F %T')] $*"; }

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      echo "model=gemma2_9b"
      if [ -f "$STATEFILE" ]; then cat "$STATEFILE"; else echo "stage=$CURRENT_STAGE"; fi
      echo "mode=gpu3_only"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits || true
    } > "$HEARTBEAT"
    sleep 60
  done
}

heartbeat_loop &
HB=$!
cleanup() { kill "$HB" >/dev/null 2>&1 || true; }
trap cleanup EXIT
write_state

wait_model() {
  CURRENT_STAGE="model_check"; write_state
  while true; do
    if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] && { [ -f "$MODEL_DIR/tokenizer.json" ] || [ -f "$MODEL_DIR/tokenizer.model" ]; }; then
      if [ -f "$MODEL_DIR/model.safetensors.index.json" ] || ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
        log "model ready: $MODEL_DIR"
        return 0
      fi
    fi
    log "model not ready: $MODEL_DIR; sleep 120s"
    sleep 120
  done
}

wait_gpu3() {
  local required="$1"
  local label="$2"
  CURRENT_STAGE="waiting_gpu3_${label}"; write_state
  while true; do
    "$PYTHON" - "$required" <<'PY'
import subprocess
import sys
required = int(sys.argv[1])
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free[int(idx)] = int(mem)
print(f"gpu3_free_mib={free.get(3, 0)} required={required}", flush=True)
raise SystemExit(0 if free.get(3, 0) >= required else 1)
PY
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "GPU3 ready for $label: >= ${required}MiB"
      return 0
    fi
    log "GPU3 not ready for $label; sleep 120s"
    sleep 120
  done
}

write_config() {
  local train_max_length="$1"
  CURRENT_STAGE="write_config_${train_max_length}"; write_state
  "$PYTHON" - "$train_max_length" <<'PY'
import json
import sys
from pathlib import Path

train_max_length = int(sys.argv[1])
base = Path("configs/strict_final_leader_soft_train64.json")
out = Path("configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json")
cfg = json.loads(base.read_text())
cfg["run_name"] = "general_base_gemma2_9b_it_coagent_conservative_train64"
cfg["resume_stages"] = True
cfg.setdefault("paths", {})["work_dir"] = "outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000/general_base_gemma2_9b_it_coagent_conservative_train64"
cfg.setdefault("model", {})["model_path"] = "/workspace/models/gemma-2-9b-it"
cfg["model"]["input_planner_adapter_path"] = None
cfg["model"]["input_coder_adapter_path"] = None
cfg["model"]["cuda_visible_devices"] = "3"
cfg["model"]["device"] = "cuda:0"
cfg.setdefault("sampling", {})["sample_limit"] = 64
cfg["sampling"]["sample_max_rounds"] = 2
cfg["sampling"]["planner_temperatures"] = [0.2, 0.5, 0.8]
cfg["sampling"]["follower_temperatures"] = [0.2, 0.5, 0.8]
cfg["sampling"]["coder_temperature"] = 0.2
cfg["sampling"]["top_p"] = 0.95
cfg.setdefault("training", {}).update({
    "leader_train_steps": 20,
    "follower_train_steps": 40,
    "train_max_samples": 256,
    "train_max_length": train_max_length,
    "learning_rate": 1e-6,
    "follower_learning_rate": 1e-6,
    "leader_learning_rate": 1e-6,
    "beta": 0.1,
    "follower_beta": 0.1,
    "leader_beta": 0.1,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "follower_lora_rank": 8,
    "follower_lora_alpha": 16,
    "follower_lora_dropout": 0.05,
    "leader_lora_rank": 8,
    "leader_lora_alpha": 16,
    "leader_lora_dropout": 0.05,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "leader_batch_size": 1,
    "leader_gradient_accumulation_steps": 8,
    "follower_batch_size": 1,
    "follower_gradient_accumulation_steps": 8,
})
cfg.setdefault("evaluation", {})["eval_split"] = "test"
cfg["evaluation"]["eval_limit"] = 17
cfg["evaluation"]["eval_max_rounds"] = 2
cfg["evaluation"]["prompt_profile"] = "legacy"
cfg["evaluation"]["best_so_far"] = True
cfg["evaluation"]["coder_adapter_start_round"] = 2
cfg.setdefault("experiment", {})["max_rounds"] = 2
cfg.setdefault("parallel_sampling", {})["enabled"] = True
cfg["parallel_sampling"]["num_shards"] = 1
cfg["parallel_sampling"]["cuda_visible_devices"] = ["3"]
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
json.loads(out.read_text())
print(out)
PY
}

cleanup_incomplete_adapters() {
  for role in follower leader; do
    if [ -d "$WORK_DIR/adapters/$role" ] && [ ! -f "$WORK_DIR/adapters/$role/adapter_model.safetensors" ]; then
      rm -rf "$WORK_DIR/adapters/$role"
    fi
  done
}

write_summary() {
  local status="$1"
  CURRENT_STAGE="summary"; write_state
  "$PYTHON" - "$status" <<'PY'
import json
import sys
from pathlib import Path
status = sys.argv[1]
summary_dir = Path("outputs/gemma2_9b_coagent_conservative_train64_gpu3_only_a6000")
work = summary_dir / "general_base_gemma2_9b_it_coagent_conservative_train64"
eval_summaries = sorted((work / "eval").glob("*summary.json"))
result = {
    "status": status,
    "config": "configs/general_base_gemma2_9b_it_coagent_conservative_train64_gpu3_only.json",
    "sampling_gpus": [3],
    "dpo_eval_gpu": 3,
    "leader_adapter_exists": (work / "adapters/leader/adapter_model.safetensors").exists(),
    "follower_adapter_exists": (work / "adapters/follower/adapter_model.safetensors").exists(),
    "eval_summary_path": str(eval_summaries[-1]) if eval_summaries else None,
}
if eval_summaries:
    result["eval_summary"] = json.loads(eval_summaries[-1].read_text())
(summary_dir / "gemma2_9b_coagent_conservative_train64_gpu3_only_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
lines = ["# Gemma2 9B CoAgent Conservative Train64 GPU3 Only", "", f"- status: {status}", f"- follower_adapter_exists: {result['follower_adapter_exists']}", f"- leader_adapter_exists: {result['leader_adapter_exists']}"]
if result["eval_summary_path"]:
    s = result["eval_summary"]
    lines += [f"- eval_summary: `{result['eval_summary_path']}`", f"- final_passed: {s.get('final_passed')}", f"- num_tasks: {s.get('num_tasks')}", f"- avg_assert_pass_rate_final: {s.get('avg_assert_pass_rate_final')}", f"- avg_total_tokens: {s.get('avg_total_tokens')}"]
(summary_dir / "gemma2_9b_coagent_conservative_train64_gpu3_only_summary.md").write_text("\n".join(lines) + "\n")
PY
}

run_once() {
  local train_max_length="$1"
  write_config "$train_max_length"
  cleanup_incomplete_adapters
  CURRENT_STAGE="strict_alternating_smoke_len_${train_max_length}"; write_state
  log "START strict-alternating-smoke config=$CONFIG train_max_length=$train_max_length gpu3_only"
  "$PYTHON" scripts/main.py strict-alternating-smoke --config "$CONFIG"
  local rc=$?
  log "END strict-alternating-smoke train_max_length=$train_max_length rc=$rc"
  return "$rc"
}

main() {
  log "============================================================"
  log "gemma2_gpu3_only_a6000"
  log "sampling_physical_gpu=3 dpo_eval_physical_gpu=3"
  wait_model
  wait_gpu3 44000 "sampling_and_dpo"
  run_once 768
  rc=$?
  if [ "$rc" -ne 0 ] && grep -R "OutOfMemoryError\\|CUDA out of memory" "$WORK_DIR/logs" >/dev/null 2>&1; then
    log "detected OOM at train_max_length=768; retry train_max_length=512"
    wait_gpu3 44000 "retry_dpo"
    run_once 512
    rc=$?
  fi
  if [ "$rc" -eq 0 ]; then
    write_summary completed
  else
    write_summary "failed_rc_${rc}"
  fi
  exit "$rc"
}

main

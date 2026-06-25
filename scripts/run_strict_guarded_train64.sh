#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="strict_guarded_train64_round2_iter0"
PARALLEL_GPUS="${PARALLEL_GPUS:-0 1 2 3}"
WAIT_FREE_MIB="${WAIT_FREE_MIB:-30000}"
LOG_DIR="$ROOT/outputs/verification_logs"
MASTER_LOG="$LOG_DIR/${TAG}_master.log"
PID_FILE="$LOG_DIR/${TAG}.pid"
LOCK_FILE="$LOG_DIR/${TAG}.lock"
HEARTBEAT="$LOG_DIR/${TAG}.heartbeat"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another ${TAG} runner is already active" >> "$MASTER_LOG"
  exit 1
fi

heartbeat_loop() {
  while true; do
    date '+%Y-%m-%d %H:%M:%S' > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$MASTER_LOG"
}

wait_for_memory() {
  local reason="$1"
  while true; do
    local all_ready=1
    local details=""
    for gpu in $PARALLEL_GPUS; do
      local free_mib
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
      details="${details} GPU${gpu}:free=${free_mib:-unknown};"
      if [[ -z "$free_mib" || "$free_mib" -lt "$WAIT_FREE_MIB" ]]; then
        all_ready=0
      fi
    done
    if [[ "$all_ready" -eq 1 ]]; then
      log "GPUs have enough memory for ${reason}: ${details}"
      return
    fi
    log "wait GPU memory for ${reason}: ${details} need free>=${WAIT_FREE_MIB}"
    sleep 300
  done
}

write_config() {
  /opt/conda/bin/python - <<'PY'
import json
import os
from pathlib import Path

base_path = Path("configs/full_algorithm_iter0_train131_round2_passonly_repaironly_step50.json")
if not base_path.exists():
    base_path = Path("configs/strict_guarded_train32_round2_iter0.json")
cfg = json.loads(base_path.read_text())

run_name = "strict_guarded_train64_round2_iter0"
parallel_gpus = os.environ.get("PARALLEL_GPUS", "0 1 2 3").split()

cfg["run_name"] = run_name
cfg["resume_stages"] = True
cfg.setdefault("paths", {})
cfg["paths"]["work_dir"] = f"outputs/{run_name}"
cfg["model"]["input_planner_adapter_path"] = None
cfg["model"]["input_coder_adapter_path"] = None
cfg["model"]["cuda_visible_devices"] = parallel_gpus[0] if parallel_gpus else "0"
cfg["model"]["device"] = "cuda:0"

cfg.setdefault("sampling", {}).update({
    "sample_split": "train",
    "sample_limit": 64,
    "sample_max_rounds": 2,
    "planner_temperatures": [0.2, 0.5, 0.8],
    "coder_temperature": 0.2,
    "top_p": 0.95,
    "seed": 17,
    "follower_limit_states": 192,
    "follower_temperatures": [0.2, 0.5, 0.8],
})
cfg.setdefault("follower_preference", {}).update({
    "require_chosen_passed": True,
    "min_chosen_pass_rate": 1.0,
    "min_pass_rate_delta": 0.1,
    "max_round1_states": 96,
    "max_repair_states": 96,
    "require_repair_improvement": True,
    "partial_chosen_weight_scale": 0.5,
})
cfg.setdefault("training", {}).update({
    "leader_train_steps": 120,
    "follower_train_steps": 180,
    "train_max_samples": 768,
    "train_max_length": 1024,
    "learning_rate": 5e-6,
    "beta": 0.1,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "leader_batch_size": 2,
    "leader_gradient_accumulation_steps": 4,
    "follower_batch_size": 2,
    "follower_gradient_accumulation_steps": 4,
})
cfg.setdefault("evaluation", {}).update({
    "eval_split": "test",
    "eval_limit": 17,
    "eval_max_rounds": 2,
    "prompt_profile": "legacy",
    "best_so_far": True,
    "coder_adapter_start_round": 2,
})
cfg.setdefault("experiment", {}).update({
    "name": "stackelberg_codepo_strict_guarded_train64",
    "seed": 17,
    "max_rounds": 2,
})
cfg.setdefault("leader_cleaning", {}).update({
    "min_chosen_pass_rate": 0.7,
    "max_pairs_per_task": 4,
    "min_response_chars": 80,
    "drop_rejected_overreach": False,
})
cfg["parallel_sampling"] = {
    "enabled": len(parallel_gpus) > 1,
    "num_shards": max(1, len(parallel_gpus)),
    "cuda_visible_devices": parallel_gpus,
}
cfg["health_gates"] = {
    "enabled": True,
    "01_follower_context_sampling": {
        "checks": {
            "trajectories": {"min": 128},
            "syntax_error_rate": {"max": 0.15},
            "positive_pass_rate_fraction": {"min": 0.10},
            "avg_rounds": {"max": 1.95}
        }
    },
    "02_follower_preference_data": {
        "checks": {
            "wpo": {"min": 16},
            "preferences": {"min": 16},
            "candidates": {"min": 48}
        }
    },
    "04_leader_sampling_after_follower": {
        "checks": {
            "trajectories": {"min": 128},
            "preferences": {"min": 20},
            "syntax_error_rate": {"max": 0.15},
            "positive_pass_rate_fraction": {"min": 0.10},
            "avg_rounds": {"max": 1.95}
        }
    },
    "06_leader_clean_data": {
        "checks": {
            "leader_clean_wpo": {"min": 20},
            "leader_round_wpo": {"min": 20},
            "leader_preferences": {"min": 20}
        }
    },
    "08_joint_eval": {
        "checks": {
            "num_tasks": {"min": 17},
            "avg_assert_pass_rate_final": {"min": 0.20}
        }
    }
}

out = Path(f"configs/{run_name}.json")
out.write_text(json.dumps(cfg, indent=2) + "\n")
print(out)
PY
}

{
  echo "============================================================"
  echo "$TAG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "root=$ROOT"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} >> "$MASTER_LOG"

write_config >> "$MASTER_LOG" 2>&1
wait_for_memory "$TAG"
log "START strict guarded train64 run"
PYTHONPATH="$ROOT/src" /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config "configs/${TAG}.json" >> "$MASTER_LOG" 2>&1
log "END strict guarded train64 run"

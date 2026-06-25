#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="strict_train131_round4_iter4"
PARALLEL_GPUS="${PARALLEL_GPUS:-0 2}"
WAIT_PARALLEL_FREE_MIB="${WAIT_PARALLEL_FREE_MIB:-38000}"
WAIT_PARALLEL_UTIL_MAX="${WAIT_PARALLEL_UTIL_MAX:-25}"
LOG_DIR="$ROOT/outputs/verification_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/${TAG}_master.log"
LOCK_FILE="$LOG_DIR/${TAG}.lock"
HEARTBEAT="$LOG_DIR/${TAG}.heartbeat"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another ${TAG} runner is already active." | tee -a "$MASTER_LOG"
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

wait_for_gpu() {
  local reason="$1"
  while true; do
    if ps -ef | grep -E 'run_train131_round4_aggressive_iter4|full_algorithm_iter[0-3]_train131_round4_aggressive|ablation_iter[0-3]_train131_round4_aggressive' | grep -v grep >/dev/null; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] wait GPUs ${PARALLEL_GPUS} for ${reason}: synchronous-control experiment still active" | tee -a "$MASTER_LOG"
      sleep 300
      continue
    fi
    local all_ready=1
    local details=""
    for gpu in $PARALLEL_GPUS; do
      local row free_mib util
      row="$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
      free_mib="${row%,*}"
      util="${row#*,}"
      details="${details} GPU${gpu}:free=${free_mib:-unknown},util=${util:-unknown};"
      if [[ -z "$free_mib" || -z "$util" || "$free_mib" -lt "$WAIT_PARALLEL_FREE_MIB" || "$util" -gt "$WAIT_PARALLEL_UTIL_MAX" ]]; then
        all_ready=0
      fi
    done
    if [[ "$all_ready" -eq 1 ]]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPUs ready for ${reason}:${details}" | tee -a "$MASTER_LOG"
      return
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] wait GPUs for ${reason}:${details} need free>=${WAIT_PARALLEL_FREE_MIB}, util<=${WAIT_PARALLEL_UTIL_MAX}" | tee -a "$MASTER_LOG"
    sleep 300
  done
}

write_configs() {
  /opt/conda/bin/python - <<'PY'
import copy
import json
import os
from pathlib import Path

base_full = json.loads(Path("configs/full_algorithm_iter0_train131_round2_passonly_repaironly_step50.json").read_text())
base_ablation = json.loads(Path("configs/ablation_iter0_train131_round2_passonly_repaironly_step50_core.json").read_text())

tag = "strict_train131_round4_iter4"
full_names = [f"strict_iter{i}_{tag}_r32_step300" for i in range(4)]
ablation_names = [f"strict_ablation_iter{i}_{tag}_r32_step300_core" for i in range(4)]
parallel_gpus = os.environ.get("PARALLEL_GPUS", "0 2").split()

def tune_full(cfg, *, run_name, work_dir, seed, sample_limit, max_rounds, follower_states, leader_steps, follower_steps, eval_limit):
    cfg["run_name"] = run_name
    cfg["resume_stages"] = True
    cfg["paths"]["work_dir"] = work_dir
    cfg["model"]["cuda_visible_devices"] = "0"
    cfg["model"]["device"] = "cuda:0"
    cfg["sampling"].update({
        "sample_split": "train",
        "sample_limit": sample_limit,
        "sample_max_rounds": max_rounds,
        "planner_temperatures": [0.2, 0.5, 0.8, 1.0],
        "coder_temperature": 0.2,
        "top_p": 0.95,
        "seed": seed,
        "follower_limit_states": follower_states,
        "follower_temperatures": [0.2, 0.5, 0.8, 1.0],
    })
    cfg["follower_preference"].update({
        "require_chosen_passed": True,
        "min_pass_rate_delta": 0.1,
        "max_round1_states": min(160, follower_states),
        "max_repair_states": max(0, follower_states - min(160, follower_states)),
    })
    cfg["training"].update({
        "leader_train_steps": leader_steps,
        "follower_train_steps": follower_steps,
        "train_max_samples": 1024,
        "train_max_length": 1536,
        "learning_rate": 8e-6,
        "beta": 0.1,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "leader_batch_size": 4,
        "leader_gradient_accumulation_steps": 2,
        "follower_batch_size": 2,
        "follower_gradient_accumulation_steps": 4,
    })
    cfg["evaluation"].update({
        "eval_split": "test",
        "eval_limit": eval_limit,
        "eval_max_rounds": max_rounds,
        "prompt_profile": "legacy",
        "best_so_far": True,
        "coder_adapter_start_round": 2,
    })
    cfg["experiment"].update({
        "name": f"stackelberg_codepo_{tag}",
        "seed": seed,
        "max_rounds": max_rounds,
    })
    cfg["generation"].update({
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 768,
        "num_samples_planner": 4,
        "num_samples_coder": 4,
    })
    cfg["leader_cleaning"].update({
        "min_chosen_pass_rate": 0.7,
        "max_pairs_per_task": 4,
        "min_response_chars": 80,
        "drop_rejected_overreach": False,
    })
    cfg["parallel_sampling"] = {
        "enabled": True,
        "num_shards": len(parallel_gpus),
        "cuda_visible_devices": parallel_gpus,
    }

smoke = copy.deepcopy(base_full)
tune_full(
    smoke,
    run_name=f"strict_smoke_{tag}",
    work_dir=f"outputs/strict_smoke_{tag}",
    seed=7,
    sample_limit=8,
    max_rounds=2,
    follower_states=24,
    leader_steps=4,
    follower_steps=4,
    eval_limit=4,
)
smoke["training"].update({
    "lora_rank": 16,
    "lora_alpha": 32,
    "train_max_length": 1024,
    "leader_batch_size": 2,
    "leader_gradient_accumulation_steps": 1,
    "follower_batch_size": 2,
    "follower_gradient_accumulation_steps": 1,
})
smoke["model"]["input_planner_adapter_path"] = None
smoke["model"]["input_coder_adapter_path"] = None
Path("configs/strict_smoke_train131_round4_iter4.json").write_text(json.dumps(smoke, indent=2) + "\n")

for i, run_name in enumerate(full_names):
    cfg = copy.deepcopy(base_full)
    tune_full(
        cfg,
        run_name=run_name,
        work_dir=f"outputs/{run_name}",
        seed=42 + i,
        sample_limit=131,
        max_rounds=4,
        follower_states=640,
        leader_steps=200,
        follower_steps=300,
        eval_limit=17,
    )
    if i == 0:
        cfg["model"]["input_planner_adapter_path"] = None
        cfg["model"]["input_coder_adapter_path"] = None
    else:
        prev = f"/workspace/multi_agent/stackelberg_codepo/outputs/{full_names[i - 1]}/adapters"
        cfg["model"]["input_planner_adapter_path"] = f"{prev}/leader"
        cfg["model"]["input_coder_adapter_path"] = f"{prev}/follower"
    Path(f"configs/{run_name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

for i, run_name in enumerate(ablation_names):
    cfg = copy.deepcopy(base_ablation)
    full_name = full_names[i]
    cfg["run_name"] = run_name
    cfg["paths"]["output_dir"] = f"outputs/{run_name}"
    cfg["paths"]["full_run_dir"] = f"outputs/{full_name}"
    cfg["model"]["cuda_visible_devices"] = "0"
    cfg["model"]["device"] = "cuda:0"
    cfg["evaluation"].update({
        "split": "test",
        "limit": 17,
        "max_rounds": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_profile": "legacy",
        "best_so_far": True,
        "coder_adapter_start_round": 2,
    })
    cfg["variants"] = [
        {"name": "prompt_only", "planner_adapter": None, "coder_adapter": None},
        {"name": "leader_only", "planner_adapter": "leader", "coder_adapter": None},
        {"name": "follower_repair_only", "planner_adapter": None, "coder_adapter": "follower"},
        {"name": "full_repair_only", "planner_adapter": "leader", "coder_adapter": "follower"},
    ]
    Path(f"configs/{run_name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

print(json.dumps({"tag": tag, "smoke": "strict_smoke_train131_round4_iter4.json", "full": full_names, "ablation": ablation_names}, indent=2))
PY
}

run_step() {
  local name="$1"
  shift
  local step_log="$LOG_DIR/${TAG}_${name}.log"
  {
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $name"
    echo "CMD: $*"
  } | tee -a "$MASTER_LOG"
  wait_for_gpu "$name"
  "$@" 2>&1 | tee -a "$step_log" "$MASTER_LOG"
  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $name"
  } | tee -a "$MASTER_LOG"
}

{
  echo "============================================================"
  echo "${TAG}"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "root=$ROOT"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} | tee -a "$MASTER_LOG"

write_configs | tee -a "$MASTER_LOG"

run_step smoke /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config configs/strict_smoke_train131_round4_iter4.json

for i in 0 1 2 3; do
  full="strict_iter${i}_${TAG}_r32_step300"
  ablation="strict_ablation_iter${i}_${TAG}_r32_step300_core"
  run_step "strict_full_iter${i}" /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config "configs/${full}.json"
  run_step "strict_ablation_iter${i}" /opt/conda/bin/python scripts/main.py run-ablation --config "configs/${ablation}.json"
done

{
  echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "outputs:"
  echo "  outputs/strict_smoke_${TAG}"
  for i in 0 1 2 3; do
    echo "  outputs/strict_iter${i}_${TAG}_r32_step300"
    echo "  outputs/strict_ablation_iter${i}_${TAG}_r32_step300_core"
  done
} | tee -a "$MASTER_LOG"

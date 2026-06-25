#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="train131_round4_aggressive_iter4"
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

write_configs() {
  /opt/conda/bin/python - <<'PY'
import copy
import json
from pathlib import Path

config_dir = Path("configs")
base_full = json.loads(Path("configs/full_algorithm_iter0_train131_round2_passonly_repaironly_step50.json").read_text())
base_ablation = json.loads(Path("configs/ablation_iter0_train131_round2_passonly_repaironly_step50_core.json").read_text())

tag = "train131_round4_aggressive_iter4"
full_names = [
    f"full_algorithm_iter{i}_{tag}_r32_step300"
    for i in range(4)
]
ablation_names = [
    f"ablation_iter{i}_{tag}_r32_step300_core"
    for i in range(4)
]

for i, run_name in enumerate(full_names):
    cfg = copy.deepcopy(base_full)
    cfg["run_name"] = run_name
    cfg["resume_stages"] = True
    cfg["paths"]["work_dir"] = f"outputs/{run_name}"
    cfg["model"]["cuda_visible_devices"] = "0"
    cfg["model"]["device"] = "cuda:0"
    if i == 0:
        cfg["model"]["input_planner_adapter_path"] = None
        cfg["model"]["input_coder_adapter_path"] = None
    else:
        prev = f"outputs/{full_names[i - 1]}/adapters"
        cfg["model"]["input_planner_adapter_path"] = f"/workspace/multi_agent/stackelberg_codepo/{prev}/leader"
        cfg["model"]["input_coder_adapter_path"] = f"/workspace/multi_agent/stackelberg_codepo/{prev}/follower"

    cfg["sampling"].update({
        "sample_split": "train",
        "sample_limit": 131,
        "sample_max_rounds": 4,
        "planner_temperatures": [0.2, 0.5, 0.8, 1.0],
        "coder_temperature": 0.2,
        "top_p": 0.95,
        "seed": 42 + i,
        "follower_limit_states": 640,
        "follower_temperatures": [0.2, 0.5, 0.8, 1.0],
    })
    cfg["follower_preference"].update({
        "require_chosen_passed": True,
        "min_pass_rate_delta": 0.1,
        "max_round1_states": 160,
        "max_repair_states": 480,
    })
    cfg["training"].update({
        "leader_train_steps": 200,
        "follower_train_steps": 300,
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
        "eval_limit": 17,
        "eval_max_rounds": 4,
        "prompt_profile": "legacy",
        "best_so_far": True,
        "coder_adapter_start_round": 2,
    })
    cfg["experiment"].update({
        "name": f"stackelberg_codepo_{tag}",
        "seed": 42 + i,
        "max_rounds": 4,
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
    Path(config_dir / f"{run_name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

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
    Path(config_dir / f"{run_name}.json").write_text(json.dumps(cfg, indent=2) + "\n")

print(json.dumps({"tag": tag, "full": full_names, "ablation": ablation_names}, indent=2))
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

for i in 0 1 2 3; do
  full="full_algorithm_iter${i}_${TAG}_r32_step300"
  ablation="ablation_iter${i}_${TAG}_r32_step300_core"
  run_step "full_iter${i}" /opt/conda/bin/python scripts/main.py full-algorithm-smoke --config "configs/${full}.json"
  run_step "ablation_iter${i}" /opt/conda/bin/python scripts/main.py run-ablation --config "configs/${ablation}.json"
done

{
  echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "outputs:"
  for i in 0 1 2 3; do
    echo "  outputs/full_algorithm_iter${i}_${TAG}_r32_step300"
    echo "  outputs/ablation_iter${i}_${TAG}_r32_step300_core"
  done
} | tee -a "$MASTER_LOG"

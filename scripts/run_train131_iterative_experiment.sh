#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOG_DIR="$ROOT/outputs/verification_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/train131_iterative_passonly_repaironly_master.log"
LOCK_FILE="$LOG_DIR/train131_iterative_passonly_repaironly.lock"
HEARTBEAT="$LOG_DIR/train131_iterative_passonly_repaironly.heartbeat"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another train131 iterative experiment runner is already active." | tee -a "$MASTER_LOG"
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

run_step() {
  local name="$1"
  shift
  local step_log="$LOG_DIR/${name}.log"
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
  echo "train131 iterative pass-only repair-only experiment"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "root=$ROOT"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
} | tee -a "$MASTER_LOG"

run_step full_iter0 /opt/conda/bin/python scripts/main.py full-algorithm-smoke --config configs/full_algorithm_iter0_train131_round2_passonly_repaironly_step50.json
run_step ablation_iter0 /opt/conda/bin/python scripts/main.py run-ablation --config configs/ablation_iter0_train131_round2_passonly_repaironly_step50_core.json
run_step full_iter1 /opt/conda/bin/python scripts/main.py full-algorithm-smoke --config configs/full_algorithm_iter1_train131_round2_passonly_repaironly_step50.json
run_step ablation_iter1 /opt/conda/bin/python scripts/main.py run-ablation --config configs/ablation_iter1_train131_round2_passonly_repaironly_step50_core.json

{
  echo "completed_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "outputs:"
  echo "  outputs/full_algorithm_iter0_train131_round2_passonly_repaironly_step50"
  echo "  outputs/ablation_iter0_train131_round2_passonly_repaironly_step50_core"
  echo "  outputs/full_algorithm_iter1_train131_round2_passonly_repaironly_step50"
  echo "  outputs/ablation_iter1_train131_round2_passonly_repaironly_step50_core"
} | tee -a "$MASTER_LOG"

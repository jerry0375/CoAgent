#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="strict_final_leader_soft"
RUNS=("strict_final_leader_soft_train64" "strict_final_leader_soft_train96")
PARALLEL_GPUS="${PARALLEL_GPUS:-0 1 2 3}"
WAIT_FREE_MIB="${WAIT_FREE_MIB:-20000}"
LOG_DIR="$ROOT/outputs/verification_logs"
MASTER_LOG="$LOG_DIR/${TAG}_master.log"
PID_FILE="$LOG_DIR/${TAG}.pid"
LOCK_FILE="$LOG_DIR/${TAG}.lock"
HEARTBEAT="$LOG_DIR/${TAG}.heartbeat"

mkdir -p "$LOG_DIR"
echo $$ > "$PID_FILE"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] another ${TAG} runner is already active" >> "$MASTER_LOG"
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
    sleep 180
  done
}

summarize_run() {
  local run="$1"
  log "SUMMARY ${run}"
  /opt/conda/bin/python - "$run" >> "$MASTER_LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
run=sys.argv[1]
root=Path('/workspace/multi_agent/stackelberg_codepo')
manifest=root/'outputs'/run/'manifest.json'
print('manifest=', manifest, 'exists=', manifest.exists())
if manifest.exists():
    m=json.load(open(manifest))
    paths=m.get('paths', {})
    for key in ['follower_wpo','leader_preferences','leader_round_wpo','leader_clean_wpo','follower_adapter','leader_adapter','eval_dir']:
        p=paths.get(key)
        if p:
            pp=Path(p)
            count=sum(1 for _ in open(pp, encoding='utf-8')) if pp.is_file() and pp.suffix=='.jsonl' else None
            print(key, p, 'exists=', pp.exists(), 'count=', count)
    eval_dir=Path(paths.get('eval_dir', root/'outputs'/run/'eval'))
    for sp in sorted(eval_dir.glob('*summary.json')):
        data=json.load(open(sp))
        keep={k:data.get(k) for k in ['num_tasks','first_round_passed','final_passed','final_pass_rate_binary','avg_assert_pass_rate_final','avg_rounds_used','avg_total_tokens','avg_total_cost','avg_leader_utility']}
        print(sp.name, json.dumps(keep, ensure_ascii=False, sort_keys=True))
PY
}

{
  echo "============================================================"
  echo "$TAG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "root=$ROOT"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  echo "runs=${RUNS[*]}"
} >> "$MASTER_LOG"

for run in "${RUNS[@]}"; do
  wait_for_memory "$run"
  log "START ${run}"
  set +e
  PYTHONPATH="$ROOT/src" /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config "configs/${run}.json" >> "$MASTER_LOG" 2>&1
  rc=$?
  set -e
  log "END ${run} returncode=${rc}"
  summarize_run "$run"
  if [[ "$rc" -ne 0 ]]; then
    log "STOP because ${run} failed; inspect ${MASTER_LOG} and outputs/${run}/logs"
    exit "$rc"
  fi
 done

log "ALL DONE ${TAG}"

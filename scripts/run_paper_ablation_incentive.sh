#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TAG="paper_ablation_incentive"
LOG_DIR="$ROOT/outputs/verification_logs"
MASTER_LOG="$LOG_DIR/${TAG}_master.log"
PID_FILE="$LOG_DIR/${TAG}.pid"
LOCK_FILE="$LOG_DIR/${TAG}.lock"
HEARTBEAT="$LOG_DIR/${TAG}.heartbeat"
SUMMARY_DIR="$ROOT/outputs/${TAG}"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"
echo $$ > "$PID_FILE"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] another ${TAG} runner is active" >> "$MASTER_LOG"
  exit 1
fi
heartbeat_loop() { while true; do date '+%Y-%m-%d %H:%M:%S' > "$HEARTBEAT"; sleep 60; done; }
heartbeat_loop & HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$MASTER_LOG"; }
wait_for_memory() {
  local reason="$1"; local gpus="${PARALLEL_GPUS:-0 1 2 3}"; local need="${WAIT_FREE_MIB:-18000}"
  while true; do
    local ok=1 details=""
    for gpu in $gpus; do
      local free_mib
      free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | head -n 1 | tr -d ' ')"
      details="${details} GPU${gpu}:free=${free_mib:-unknown};"
      if [[ -z "$free_mib" || "$free_mib" -lt "$need" ]]; then ok=0; fi
    done
    if [[ "$ok" -eq 1 ]]; then log "GPUs have enough memory for ${reason}: ${details}"; return; fi
    log "wait GPU memory for ${reason}: ${details} need free>=${need}"; sleep 180
  done
}
run_eval() {
  local name="$1" planner="$2" coder="$3"
  log "START eval ${name}"
  local outdir="$ROOT/outputs/${TAG}/${name}"
  mkdir -p "$outdir"
  local cmd=(/opt/conda/bin/python -m stackelberg_codepo.evaluation.role_eval
    --config configs/strict_final_leader_soft_train64.json
    --model-path /workspace/models/Qwen2.5-Coder-1.5B-Instruct
    --split test --limit 17 --output-dir "$outdir" --output-name "$name"
    --device cuda:0 --max-rounds 2 --temperature 0.0 --top-p 1.0 --prompt-profile legacy
    --best-so-far --coder-adapter-start-round 2 --no-resume)
  if [[ -n "$planner" ]]; then cmd+=(--planner-adapter-path "$planner"); fi
  if [[ -n "$coder" ]]; then cmd+=(--coder-adapter-path "$coder"); fi
  "${cmd[@]}" >> "$MASTER_LOG" 2>&1
  log "END eval ${name} rc=$?"
}
run_full() {
  local run="$1"
  wait_for_memory "$run"
  log "START full ${run}"
  set +e
  PYTHONPATH="$ROOT/src" /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config "configs/${run}.json" >> "$MASTER_LOG" 2>&1
  local rc=$?
  set -e
  log "END full ${run} rc=${rc}"
  if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
}
write_summary() {
  log "START summary"
  /opt/conda/bin/python - <<'PY' >> "$MASTER_LOG" 2>&1
import csv, json
from pathlib import Path
root=Path('/workspace/multi_agent/stackelberg_codepo')
out=root/'outputs/paper_ablation_incentive'
rows=[]
def add(method, source, summary_path):
    d=json.load(open(summary_path))
    rows.append({
        'Method': method,
        'Source': source,
        'Pass Rate (%)': round(100*float(d.get('final_pass_rate_binary', 0.0)), 2),
        'Assert Pass Rate (%)': round(100*float(d.get('avg_assert_pass_rate_final', 0.0)), 2),
        'Avg. Rounds': round(float(d.get('avg_rounds_used', 0.0)), 2),
        'Avg. Tokens': round(float(d.get('avg_total_tokens', 0.0))),
        'final_passed': d.get('final_passed'),
        'num_tasks': d.get('num_tasks'),
    })
add('Prompt-only', 'paper_table_recheck_prompt_only', out/'paper_table_recheck_prompt_only/paper_table_recheck_prompt_only_summary.json')
add('Follower-only', 'paper_table_recheck_follower_only', out/'paper_table_recheck_follower_only/paper_table_recheck_follower_only_summary.json')
for method, run in [
    ('CoAgent w/o Incentive', 'strict_final_leader_soft_no_incentive_train64'),
    ('CoAgent + Shuffled Incentive', 'strict_final_leader_soft_shuffled_incentive_train64'),
]:
    add(method, run, root/'outputs'/run/'eval'/f'{run}_leader_follower_eval_summary.json')
json_path=out/'paper_ablation_incentive_summary.json'
md_path=out/'paper_ablation_incentive_summary.md'
csv_path=out/'paper_ablation_incentive_summary.csv'
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
headers=['Method','Pass Rate (%)','Assert Pass Rate (%)','Avg. Rounds','Avg. Tokens','Source']
lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
for r in rows:
    lines.append('| '+' | '.join(str(r[h]) for h in headers)+' |')
md_path.write_text('\n'.join(lines)+'\n')
print(md_path.read_text())
print('summary_json=', json_path)
print('summary_csv=', csv_path)
PY
  log "END summary"
}
{
  echo "============================================================"
  echo "$TAG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} >> "$MASTER_LOG"
wait_for_memory recheck_eval
run_eval paper_table_recheck_prompt_only "" ""
run_eval paper_table_recheck_follower_only "" "$ROOT/outputs/strict_final_leader_soft_train64/adapters/follower"
run_full strict_final_leader_soft_no_incentive_train64
run_full strict_final_leader_soft_shuffled_incentive_train64
write_summary
log "ALL DONE ${TAG}"

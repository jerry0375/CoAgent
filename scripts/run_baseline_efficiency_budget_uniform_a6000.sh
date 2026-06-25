#!/usr/bin/env bash
set -u
ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1
source "$ROOT/proxy_env.sh" 2>/dev/null || true
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export HF_HOME="/workspace/.cache/huggingface"
export HF_HUB_DISABLE_XET=1

OUT_DIR="$ROOT/outputs/baseline_efficiency_budget_uniform_a6000"
LOG_DIR="$ROOT/outputs/verification_logs"
MASTER_LOG="$LOG_DIR/baseline_efficiency_budget_uniform_a6000_master.log"
PID_FILE="$LOG_DIR/baseline_efficiency_budget_uniform_a6000.pid"
HEARTBEAT="$LOG_DIR/baseline_efficiency_budget_uniform_a6000.heartbeat"
MODEL="/workspace/models/Qwen2.5-Coder-1.5B-Instruct"
CONFIG="$ROOT/configs/strict_final_leader_soft_train64.json"
METHODS=(direct_prompt self_consistency_3 self_repair agentcoder)
GPUS=(0 1 2 3)
# 16 total-budget points, uniformly covering roughly 0-3000 while avoiding unusable near-zero generations.
BUDGETS=(256 439 622 805 988 1171 1354 1537 1720 1903 2086 2269 2452 2635 2818 3000)
mkdir -p "$OUT_DIR" "$LOG_DIR"
echo $$ > "$PID_FILE"
exec >> "$MASTER_LOG" 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      echo "stage=${CURRENT_STAGE:-starting}"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null || true
      echo "workers:"
      jobs -l 2>/dev/null || true
    } > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HB_PID=$!
trap 'kill "$HB_PID" >/dev/null 2>&1 || true' EXIT

alloc_tokens() {
  local method="$1" budget="$2"
  /opt/conda/bin/python - <<PY
method='$method'
budget=int('$budget')
# max_new_tokens is the per-code-generation budget consumed by paper_main_baselines.py.
# plan_max_new_tokens is only used by AgentCoder's tester.
if method == 'direct_prompt':
    max_new=budget
    plan=64
elif method == 'self_consistency_3':
    max_new=max(64, budget // 3)
    plan=64
elif method == 'self_repair':
    max_new=max(64, budget // 2)
    plan=64
elif method == 'agentcoder':
    # Worst-case path is code + tester + repair. Reserve about 20% for tester, capped at 384,
    # then split the rest across initial code and repair.
    plan=max(64, min(384, round(budget * 0.20)))
    max_new=max(64, (budget - plan) // 2)
else:
    max_new=budget
    plan=64
print(f'{max_new} {plan}')
PY
}

summarize() {
  /opt/conda/bin/python - <<'PY'
import csv, json
from pathlib import Path
root=Path('/workspace/multi_agent/stackelberg_codepo/outputs/baseline_efficiency_budget_uniform_a6000')
method_display={
 'direct_prompt':'Direct Prompt',
 'self_consistency_3':'Self-Consistency@3',
 'self_repair':'Self-Repair',
 'agentcoder':'AgentCoder',
}
rows=[]
for d in sorted(root.glob('*_budget*')):
    if not d.is_dir():
        continue
    meta_path=d/'meta.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    method=meta.get('method')
    summary_path=d/method/f'{method}_summary.json' if method else d/'missing_summary.json'
    row={
      'method':method_display.get(method, method),
      'method_key':method,
      'declared_total_budget_tokens':meta.get('declared_total_budget_tokens'),
      'max_new_tokens':meta.get('max_new_tokens'),
      'plan_max_new_tokens':meta.get('plan_max_new_tokens'),
      'budget_allocation':meta.get('budget_allocation'),
      'final_passed':None,
      'num_tasks':None,
      'pass_rate_percent':None,
      'assert_pass_rate_percent':None,
      'avg_total_tokens_actual':None,
      'status':'pending',
      'summary_path':str(summary_path) if summary_path.exists() else '',
    }
    if summary_path.exists():
        try:
            s=json.loads(summary_path.read_text(encoding='utf-8'))
            row.update({
              'final_passed':s.get('final_passed'),
              'num_tasks':s.get('num_tasks'),
              'pass_rate_percent':round(100*float(s.get('pass_rate',0.0)),2),
              'assert_pass_rate_percent':round(100*float(s.get('avg_assert_pass_rate',0.0)),2),
              'avg_total_tokens_actual':s.get('avg_total_tokens'),
              'status':'completed',
            })
        except Exception as exc:
            row['status']='summary_parse_failed'; row['error']=repr(exc)
    fail_path=d/'FAILED.json'
    if fail_path.exists():
        row['status']='failed'
        try: row['error']=json.loads(fail_path.read_text(encoding='utf-8')).get('returncode')
        except Exception: row['error']='failed'
    rows.append(row)
order={m:i for i,m in enumerate(['direct_prompt','self_consistency_3','self_repair','agentcoder'])}
rows.sort(key=lambda r:(order.get(r.get('method_key'),99), int(r.get('declared_total_budget_tokens') or 0)))
fields=['method','method_key','declared_total_budget_tokens','max_new_tokens','plan_max_new_tokens','budget_allocation','final_passed','num_tasks','pass_rate_percent','assert_pass_rate_percent','avg_total_tokens_actual','status','summary_path']
(root/'baseline_efficiency_budget_uniform_summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with (root/'baseline_efficiency_budget_uniform_summary.csv').open('w', encoding='utf-8', newline='') as f:
    w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
with (root/'baseline_efficiency_budget_uniform_summary.md').open('w', encoding='utf-8') as f:
    f.write('| method | budget | max_new | plan_max | final_passed | num_tasks | Pass Rate (%) | Assert Pass Rate (%) | Actual Avg Tokens | status |\n')
    f.write('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n')
    for r in rows:
        def fmt(v, nd=2):
            if v is None: return ''
            try: return f'{float(v):.{nd}f}'
            except Exception: return str(v)
        f.write(f"| {r['method']} | {r.get('declared_total_budget_tokens') or ''} | {r.get('max_new_tokens') or ''} | {r.get('plan_max_new_tokens') or ''} | {r.get('final_passed') if r.get('final_passed') is not None else ''} | {r.get('num_tasks') if r.get('num_tasks') is not None else ''} | {fmt(r.get('pass_rate_percent'))} | {fmt(r.get('assert_pass_rate_percent'))} | {fmt(r.get('avg_total_tokens_actual'),0)} | {r.get('status')} |\n")
print((root/'baseline_efficiency_budget_uniform_summary.md').read_text(encoding='utf-8'))
PY
}

validate_inputs() {
  CURRENT_STAGE="validate_inputs"
  local rc=0
  for p in "$MODEL/config.json" "$CONFIG" "$ROOT/scripts/paper_main_baselines.py"; do
    if [ ! -f "$p" ]; then log "ERROR missing $p"; rc=1; fi
  done
  /opt/conda/bin/python scripts/paper_main_baselines.py --help >/dev/null || rc=1
  /opt/conda/bin/python -m py_compile scripts/paper_main_baselines.py || rc=1
  return "$rc"
}

run_point() {
  local method="$1" budget="$2" gpu="$3"
  local alloc max_new plan point point_dir summary
  alloc=$(alloc_tokens "$method" "$budget") || return 1
  max_new=$(echo "$alloc" | awk '{print $1}')
  plan=$(echo "$alloc" | awk '{print $2}')
  point="${method}_budget${budget}"
  point_dir="$OUT_DIR/$point"
  summary="$point_dir/$method/${method}_summary.json"
  mkdir -p "$point_dir"
  /opt/conda/bin/python - <<PY
import json
from pathlib import Path
method='$method'; budget=int('$budget'); max_new=int('$max_new'); plan=int('$plan')
if method == 'direct_prompt': allocation='single_generation_budget=B'
elif method == 'self_consistency_3': allocation='three_samples_each_floor(B/3)'
elif method == 'self_repair': allocation='initial_and_repair_each_floor(B/2)'
elif method == 'agentcoder': allocation='code_and_repair_share_B_minus_tester_plan; tester_plan_scaled_with_B'
else: allocation='unknown'
Path('$point_dir/meta.json').write_text(json.dumps({
  'method':method,
  'declared_total_budget_tokens':budget,
  'max_new_tokens':max_new,
  'plan_max_new_tokens':plan,
  'budget_allocation':allocation,
  'gpu':int('$gpu'),
  'model':'$MODEL',
  'split':'test',
  'limit':17,
}, indent=2)+'\n', encoding='utf-8')
PY
  if [ -f "$summary" ]; then
    if /opt/conda/bin/python - <<PY
import json, sys
s=json.load(open('$summary'))
ok=s.get('num_tasks') == 17 and float(s.get('avg_total_tokens',0)) > 0 and 0 <= float(s.get('pass_rate',-1)) <= 1 and 0 <= float(s.get('avg_assert_pass_rate',-1)) <= 1
sys.exit(0 if ok else 1)
PY
    then
      log "SKIP completed $point"
      summarize
      return 0
    fi
    log "existing invalid summary, rerun $point"
    rm -rf "$point_dir/$method" "$point_dir/main_baselines_summary.json" "$point_dir/main_baselines_summary.md"
  fi
  log "START $point on gpu=$gpu total_budget=$budget max_new=$max_new plan_max=$plan"
  CUDA_VISIBLE_DEVICES="$gpu" /opt/conda/bin/python scripts/paper_main_baselines.py \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --split test \
    --limit 17 \
    --output-dir "$point_dir" \
    --device cuda:0 \
    --methods "$method" \
    --max-new-tokens "$max_new" \
    --plan-max-new-tokens "$plan" \
    --temperature 0.2 \
    --sample-temperature 0.7 \
    --top-p 0.95 \
    --no-resume
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    log "FAILED $point rc=$rc"
    /opt/conda/bin/python - <<PY
import json
from pathlib import Path
Path('$point_dir/FAILED.json').write_text(json.dumps({'point':'$point','method':'$method','declared_total_budget_tokens':$budget,'max_new_tokens':$max_new,'plan_max_new_tokens':$plan,'gpu':$gpu,'returncode':$rc}, indent=2)+'\n')
PY
    summarize || true
    return "$rc"
  fi
  /opt/conda/bin/python - <<PY
import json
s=json.load(open('$summary'))
assert s.get('num_tasks') == 17, s.get('num_tasks')
assert float(s.get('avg_total_tokens',0)) > 0
assert 0 <= float(s.get('pass_rate',-1)) <= 1
assert 0 <= float(s.get('avg_assert_pass_rate',-1)) <= 1
PY
  log "DONE $point"
  summarize
}

worker() {
  local method="$1" gpu="$2"
  log "worker start method=$method gpu=$gpu"
  for budget in "${BUDGETS[@]}"; do
    run_point "$method" "$budget" "$gpu" || return $?
  done
  log "worker done method=$method gpu=$gpu"
}

main() {
  log "============================================================"
  log "baseline_efficiency_budget_uniform_a6000 started"
  hostname
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader || true
  validate_inputs || { log "input validation failed"; exit 2; }
  summarize || true
  CURRENT_STAGE="workers_running"
  pids=()
  for i in 0 1 2 3; do
    worker "${METHODS[$i]}" "${GPUS[$i]}" &
    pids+=("$!")
  done
  rc=0
  for p in "${pids[@]}"; do
    if ! wait "$p"; then rc=1; fi
  done
  CURRENT_STAGE="finished"
  summarize || true
  if [ "$rc" -ne 0 ]; then log "finished with failures"; exit "$rc"; fi
  log "baseline_efficiency_budget_uniform_a6000 finished successfully"
}
main

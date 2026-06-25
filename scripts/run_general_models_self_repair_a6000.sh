#!/usr/bin/env bash
set -u

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/outputs/verification_logs"
OUT_DIR="$ROOT/outputs/general_models_self_repair_a6000"
LOG="$LOG_DIR/general_models_self_repair_a6000_master.log"
PIDFILE="$LOG_DIR/general_models_self_repair_a6000.pid"
HEARTBEAT="$LOG_DIR/general_models_self_repair_a6000.heartbeat"
mkdir -p "$LOG_DIR" "$OUT_DIR"
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

wait_gpu_free() {
  local gpu="$1"
  local need="$2"
  local label="$3"
  CURRENT_STAGE="waiting_${label}_gpu${gpu}"
  while true; do
    "$PYTHON" - "$gpu" "$need" <<'PY'
import subprocess, sys
gpu = int(sys.argv[1]); need = int(sys.argv[2])
out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
free = {}
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free[int(idx)] = int(mem)
print(f"gpu{gpu}_free_mib={free.get(gpu, 0)} required={need}", flush=True)
raise SystemExit(0 if free.get(gpu, 0) >= need else 1)
PY
    if [ "$?" -eq 0 ]; then
      log "GPU${gpu} ready for ${label}"
      return 0
    fi
    log "GPU${gpu} does not have enough free memory for ${label}; sleep 120s"
    sleep 120
  done
}

run_model() {
  local gpu="$1"
  local label="$2"
  local model_path="$3"
  local min_free="$4"
  local model_out="$OUT_DIR/$label"
  local worker_log="$LOG_DIR/general_models_self_repair_${label}.log"
  local worker_pid="$LOG_DIR/general_models_self_repair_${label}.pid"
  if [ ! -d "$model_path" ] || [ ! -f "$model_path/config.json" ]; then
    log "SKIP $label: missing model path $model_path"
    cat > "$model_out.MISSING.json" <<EOF
{"label":"$label","status":"missing_model","model_path":"$model_path"}
EOF
    return 0
  fi
  if [ -f "$model_out/self_repair/self_repair_summary.json" ]; then
    local n
    n=$("$PYTHON" - "$model_out/self_repair/self_repair_summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("num_tasks", 0))
PY
)
    if [ "$n" = "17" ]; then
      log "SKIP $label: existing completed summary"
      return 0
    fi
  fi
  wait_gpu_free "$gpu" "$min_free" "$label"
  (
    set -u
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
    cd "$ROOT" || exit 1
    echo "[$(date '+%F %T')] START $label gpu=$gpu model=$model_path"
    "$PYTHON" scripts/paper_main_baselines.py \
      --config configs/strict_final_leader_soft_train64.json \
      --model-path "$model_path" \
      --split test \
      --limit 17 \
      --output-dir "$model_out" \
      --device cuda:0 \
      --methods self_repair \
      --max-new-tokens 512 \
      --plan-max-new-tokens 192 \
      --temperature 0.2 \
      --sample-temperature 0.7 \
      --top-p 0.95
    rc=$?
    echo "[$(date '+%F %T')] END $label rc=$rc"
    exit "$rc"
  ) > "$worker_log" 2>&1 &
  echo $! > "$worker_pid"
  log "started $label pid=$! gpu=$gpu log=$worker_log"
}

summarize() {
  CURRENT_STAGE="summarize"
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("outputs/general_models_self_repair_a6000")
rows = []
for label, display in [
    ("llama31_8b_instruct", "Meta-Llama-3.1-8B-Instruct"),
    ("gemma2_9b_it", "gemma-2-9b-it"),
    ("mistral_7b_instruct_v03", "Mistral-7B-Instruct-v0.3"),
]:
    summary = root / label / "self_repair" / "self_repair_summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        rows.append({
            "Model": display,
            "Method": "Self-Repair",
            "Pass Rate (%)": round(100*s.get("pass_rate", 0), 2),
            "Assert Pass Rate (%)": round(100*s.get("avg_assert_pass_rate", 0), 2),
            "Avg. Tokens": round(s.get("avg_total_tokens", 0)),
            "final_passed": s.get("final_passed"),
            "num_tasks": s.get("num_tasks"),
            "status": "completed",
            "summary_path": str(summary),
        })
    else:
        missing = root / f"{label}.MISSING.json"
        rows.append({"Model": display, "Method": "Self-Repair", "status": "missing_model" if missing.exists() else "running_or_failed", "summary_path": str(summary)})
(root / "self_repair_summary_all.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2)+"\n")
lines = ["| Model | Method | Pass Rate (%) | Assert Pass Rate (%) | Avg. Tokens | final_passed | num_tasks | status |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
for r in rows:
    lines.append(f"| {r.get('Model')} | {r.get('Method')} | {r.get('Pass Rate (%)','')} | {r.get('Assert Pass Rate (%)','')} | {r.get('Avg. Tokens','')} | {r.get('final_passed','')} | {r.get('num_tasks','')} | {r.get('status')} |")
(root / "self_repair_summary_all.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
PY
}

main() {
  log "============================================================"
  log "START general model Self-Repair baselines"
  CURRENT_STAGE="launch_workers"
  run_model 2 "mistral_7b_instruct_v03" "/workspace/models/Mistral-7B-Instruct-v0.3" 16000
  run_model 3 "gemma2_9b_it" "/workspace/models/gemma-2-9b-it" 22000
  # Llama is included for completeness. On this A6000 server the model may be absent; if so it is recorded without blocking Gemma/Mistral.
  run_model 0 "llama31_8b_instruct" "/workspace/models/Meta-Llama-3.1-8B-Instruct" 22000

  CURRENT_STAGE="waiting_workers"
  while true; do
    alive=0
    for f in "$LOG_DIR"/general_models_self_repair_*.pid; do
      [ -f "$f" ] || continue
      pid=$(cat "$f" 2>/dev/null || true)
      [ -n "$pid" ] || continue
      if kill -0 "$pid" >/dev/null 2>&1; then
        alive=$((alive+1))
      fi
    done
    summarize || true
    [ "$alive" -eq 0 ] && break
    sleep 120
  done
  summarize
  log "DONE general model Self-Repair baselines"
}

main "$@"

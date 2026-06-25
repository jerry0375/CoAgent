#!/usr/bin/env bash
set -euo pipefail
ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"
source "$ROOT/proxy_env.sh" 2>/dev/null || true
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_XET=1
export HF_HOME="/workspace/.cache/huggingface"
OUT_DIR="$ROOT/outputs/qwen32_humaneval_direct_recheck_a800"
LOG_DIR="$ROOT/outputs/verification_logs"
MASTER_LOG="$LOG_DIR/qwen32_humaneval_direct_recheck_a800_master.log"
PID_FILE="$LOG_DIR/qwen32_humaneval_direct_recheck_a800.pid"
HEARTBEAT="$LOG_DIR/qwen32_humaneval_direct_recheck_a800.heartbeat"
MODEL="/workspace/models/Qwen2.5-Coder-32B-Instruct"
mkdir -p "$OUT_DIR" "$LOG_DIR"
echo $$ > "$PID_FILE"
heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader || true
    } > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT
{
  echo "============================================================"
  echo "qwen32_humaneval_direct_recheck_a800"
  echo "started_at=$(date '+%F %T')"
  hostname
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader || true
  /opt/conda/bin/python - <<'PY'
from pathlib import Path
p=Path('/workspace/models/Qwen2.5-Coder-32B-Instruct')
expected=[p/f'model-{i:05d}-of-00014.safetensors' for i in range(1,15)]
missing=[str(x) for x in expected if not x.exists()]
print('model_complete=', (p/'config.json').exists() and (p/'tokenizer.json').exists() and not missing)
if missing:
    print('missing=', missing)
    raise SystemExit(2)
PY
  /opt/conda/bin/python scripts/paper_main_baselines.py \
    --config configs/strict_final_leader_soft_train64.json \
    --model-path "$MODEL" \
    --split test \
    --limit 17 \
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --methods direct_prompt \
    --temperature 0.2 \
    --sample-temperature 0.7 \
    --top-p 0.95 \
    --no-resume
  echo "finished_at=$(date '+%F %T')"
  echo "summary=$OUT_DIR/direct_prompt/direct_prompt_summary.json"
  cat "$OUT_DIR/direct_prompt/direct_prompt_summary.json"
} >> "$MASTER_LOG" 2>&1

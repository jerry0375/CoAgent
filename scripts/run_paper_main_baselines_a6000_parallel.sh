#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
mkdir -p outputs/verification_logs outputs/paper_main_baselines_a6000_parallel

/opt/conda/bin/python -m py_compile scripts/paper_main_baselines.py

start_method() {
  local gpu="$1"
  local method="$2"
  local out_dir="outputs/paper_main_baselines_a6000_parallel/${method}"
  local log="outputs/verification_logs/paper_main_${method}_a6000.log"
  local pid_file="outputs/verification_logs/paper_main_${method}_a6000.pid"

  if ps -eo cmd | grep -F "paper_main_baselines.py" | grep -F -- "--methods ${method}" | grep -v grep >/dev/null; then
    echo "${method} already running"
    return
  fi

  if [ -s "${out_dir}/${method}/${method}.jsonl" ]; then
    local lines
    lines="$(wc -l < "${out_dir}/${method}/${method}.jsonl")"
    if [ "$lines" -ge 17 ]; then
      echo "${method} already complete (${lines} records)"
      return
    fi
  fi

  nohup env PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES="$gpu" /opt/conda/bin/python scripts/paper_main_baselines.py \
    --config configs/strict_final_leader_soft_train64.json \
    --model-path /workspace/models/Qwen2.5-Coder-1.5B-Instruct \
    --split test \
    --limit 17 \
    --output-dir "$out_dir" \
    --device cuda:0 \
    --methods "$method" \
    --temperature 0.2 \
    --sample-temperature 0.7 \
    --top-p 0.95 \
    > "$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "started ${method} pid=$! gpu=${gpu} log=${log}"
}

start_method 1 self_consistency_3
start_method 2 mad
start_method 3 agentcoder

ps -eo pid,ppid,stat,etime,cmd | grep -E 'paper_main_baselines.py' | grep -v grep || true

#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export http_proxy=http://118.202.46.217:7897
export https_proxy=http://118.202.46.217:7897
export HTTP_PROXY=http://118.202.46.217:7897
export HTTPS_PROXY=http://118.202.46.217:7897
export all_proxy=socks5h://118.202.46.217:7897
export ALL_PROXY=socks5h://118.202.46.217:7897
export no_proxy=localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8
export NO_PROXY=localhost,127.0.0.1,::1,192.168.0.0/16,10.0.0.0/8
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

MODEL_ROOT="/workspace/models"
OUT_DIR="$ROOT/outputs/model_capacity_direct_a6000"
LOG_DIR="$ROOT/outputs/verification_logs"
mkdir -p "$MODEL_ROOT" "$OUT_DIR" "$LOG_DIR"

/opt/conda/bin/python -m py_compile scripts/paper_main_baselines.py

download_model() {
  local model_id="$1"
  local target="$2"
  echo "[download/resume] $model_id -> $target"
  /opt/conda/bin/python - "$model_id" "$target" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_id, target = sys.argv[1], sys.argv[2]
Path(target).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=model_id,
    local_dir=target,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

start_model() {
  local gpu="$1"
  local label="$2"
  local hf_id="$3"
  local local_path="$4"
  local log="$LOG_DIR/model_capacity_${label}_a6000.log"
  local pid_file="$LOG_DIR/model_capacity_${label}_a6000.pid"
  local heartbeat="$LOG_DIR/model_capacity_${label}_a6000.heartbeat"

  if ps -eo cmd | grep -F "model_capacity_direct_a6000_worker" | grep -F "$label" | grep -v grep >/dev/null; then
    echo "$label already running"
    return
  fi

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    while true; do date '+%F %T' > "$heartbeat"; sleep 60; done &
    hb=$!
    trap 'kill "$hb" 2>/dev/null || true' EXIT
    echo "model_capacity_direct_a6000_worker $label"
    echo "started_at=$(date '+%F %T') gpu=$gpu"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader || true
    download_model "$hf_id" "$local_path"
    /opt/conda/bin/python scripts/paper_main_baselines.py \
      --config configs/strict_final_leader_soft_train64.json \
      --model-path "$local_path" \
      --split test \
      --limit 17 \
      --output-dir "$OUT_DIR/$label" \
      --device cuda:0 \
      --methods direct_prompt \
      --temperature 0.2 \
      --sample-temperature 0.7 \
      --top-p 0.95
    echo "finished_at=$(date '+%F %T')"
  ) > "$log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "started $label pid=$! gpu=$gpu log=$log"
}

start_model 0 qwen25_coder_3b_instruct Qwen/Qwen2.5-Coder-3B-Instruct "$MODEL_ROOT/Qwen2.5-Coder-3B-Instruct"
start_model 1 qwen25_coder_7b_instruct Qwen/Qwen2.5-Coder-7B-Instruct "$MODEL_ROOT/Qwen2.5-Coder-7B-Instruct"

ps -eo pid,ppid,stat,etime,cmd | grep -E 'paper_main_baselines.py|model_capacity_direct_a6000_worker' | grep -v grep || true

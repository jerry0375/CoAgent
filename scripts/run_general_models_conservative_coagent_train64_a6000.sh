#!/usr/bin/env bash
set -u

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1

LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/general_models_conservative_coagent_train64_a6000_master.log"
PIDFILE="$LOG_DIR/general_models_conservative_coagent_train64_a6000.pid"
HEARTBEAT="$LOG_DIR/general_models_conservative_coagent_train64_a6000.heartbeat"
STATEFILE="$LOG_DIR/general_models_conservative_coagent_train64_a6000.state"
LOCKFILE="$LOG_DIR/general_models_conservative_coagent_train64_a6000.lock"

mkdir -p "$LOG_DIR"

exec >> "$LOG" 2>&1

if [ -f "$LOCKFILE" ]; then
  old_pid="$(cat "$LOCKFILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] existing runner still active: pid=$old_pid"
    exit 0
  fi
fi
echo $$ > "$LOCKFILE"
echo $$ > "$PIDFILE"

CURRENT_MODEL="none"
CURRENT_STAGE="starting"

write_state() {
  {
    echo "model=$CURRENT_MODEL"
    echo "stage=$CURRENT_STAGE"
  } > "$STATEFILE"
}

log() {
  echo "[$(date '+%F %T')] $*"
}

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      if [ -f "$STATEFILE" ]; then
        cat "$STATEFILE"
      else
        echo "model=$CURRENT_MODEL"
        echo "stage=$CURRENT_STAGE"
      fi
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits || true
      fi
    } > "$HEARTBEAT"
    sleep 60
  done
}

heartbeat_loop &
HB_PID=$!

cleanup() {
  kill "$HB_PID" >/dev/null 2>&1 || true
  rm -f "$LOCKFILE"
}
trap cleanup EXIT
write_state

PYTHON="/opt/conda/bin/python"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REMOTE_HOST="219.216.65.85"
REMOTE_PORT="32791"
REMOTE_PASSWORD="${REMOTE_PASSWORD:-}"

validate_model_dir() {
  local model_dir="$1"
  "$PYTHON" - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
if not model_dir.is_dir():
    raise SystemExit(1)
if not (model_dir / "config.json").is_file():
    raise SystemExit(1)
if not ((model_dir / "tokenizer.json").is_file() or (model_dir / "tokenizer.model").is_file()):
    raise SystemExit(1)
if list(model_dir.glob("*.incomplete")) or list(model_dir.glob("*.partial")):
    raise SystemExit(1)
index = model_dir / "model.safetensors.index.json"
if index.is_file():
    data = json.loads(index.read_text())
    files = sorted(set(data.get("weight_map", {}).values()))
    if not files or any(not (model_dir / name).is_file() for name in files):
        raise SystemExit(1)
    raise SystemExit(0)
if list(model_dir.glob("*.safetensors")) or list(model_dir.glob("*.bin")):
    raise SystemExit(0)
raise SystemExit(1)
PY
}

scp_with_password() {
  local remote_path="$1"
  local transfer_dir="$2"
  REMOTE_PATH="$remote_path" TRANSFER_DIR="$transfer_dir" "$PYTHON" - <<'PY'
import errno
import os
import pty
import select
import sys
import time

password = os.environ.get("REMOTE_PASSWORD")
if not password:
    print("REMOTE_PASSWORD is required for password-based transfer", file=sys.stderr)
    raise SystemExit(2)
remote = os.environ["REMOTE_PATH"]
target = os.environ["TRANSFER_DIR"]
port = os.environ.get("REMOTE_PORT", "32791")
cmd = [
    "scp", "-r", "-P", port,
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/root/.ssh/known_hosts",
    f"root@{os.environ.get('REMOTE_HOST', '219.216.65.85')}:{remote}",
    target,
]
print("server_to_server_cmd=" + " ".join(cmd), flush=True)
pid, fd = pty.fork()
if pid == 0:
    os.execvp(cmd[0], cmd)
sent = False
status = 1
while True:
    try:
        readable, _, _ = select.select([fd], [], [], 1.0)
    except OSError:
        readable = []
    if fd in readable:
        try:
            data = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                data = b""
            else:
                raise
        if data:
            text = data.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            low = text.lower()
            if "are you sure" in low:
                os.write(fd, b"yes\n")
            if "password:" in low and not sent:
                os.write(fd, (password + "\n").encode())
                sent = True
    done_pid, done_status = os.waitpid(pid, os.WNOHANG)
    if done_pid == pid:
        status = os.WEXITSTATUS(done_status) if os.WIFEXITED(done_status) else 128
        break
    time.sleep(0.05)
raise SystemExit(status)
PY
}

transfer_model_if_needed() {
  local label="$1"
  local local_dir="$2"
  local remote_dir="$3"
  local transfer_dir="${local_dir}.transfer"
  CURRENT_MODEL="$label"
  CURRENT_STAGE="model_check"
  write_state
  if validate_model_dir "$local_dir"; then
    log "$label model already valid: $local_dir"
    return 0
  fi
  log "$label model missing or incomplete on A6000; transfer from A800"
  mkdir -p /workspace/models
  for attempt in 1 2 3; do
    CURRENT_STAGE="model_transfer_${label}_attempt_${attempt}"
    write_state
    log "$label transfer attempt $attempt/3"
    rm -rf "$transfer_dir"
    export REMOTE_HOST REMOTE_PORT REMOTE_PASSWORD
    scp_with_password "$remote_dir" "$transfer_dir"
    rc=$?
    log "$label scp exited rc=$rc"
    if validate_model_dir "$transfer_dir"; then
      if [ -d "$local_dir" ]; then
        mv "$local_dir" "${local_dir}.incomplete.$(date +%s)"
      fi
      mv "$transfer_dir" "$local_dir"
      log "$label model installed: $local_dir"
      return 0
    fi
    log "$label transferred model failed validation; retry in 60s"
    sleep 60
  done
  log "ERROR: $label model transfer failed after 3 attempts"
  return 1
}

wait_sampling_gpus() {
  CURRENT_STAGE="waiting_for_sampling_gpu_memory"
  write_state
  while true; do
    "$PYTHON" - <<'PY'
import subprocess
import sys

out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
free = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free.append((int(idx), int(mem)))
print("sampling_gpu_free_mib=" + ",".join(f"{idx}:{mem}" for idx, mem in free), flush=True)
raise SystemExit(0 if len(free) >= 4 and all(mem >= 30000 for _, mem in free[:4]) else 1)
PY
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "sampling GPU memory ready: all first four GPUs >= 30000MiB"
      return 0
    fi
    log "sampling GPU memory not ready; sleep 120s"
    sleep 120
  done
}

choose_dpo_gpu() {
  CURRENT_STAGE="waiting_for_dpo_gpu_memory"
  write_state
  while true; do
    best="$("$PYTHON" - <<'PY'
import subprocess
import sys

out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
free = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(",")]
    free.append((int(idx), int(mem)))
print("dpo_gpu_free_mib=" + ",".join(f"{idx}:{mem}" for idx, mem in free), file=sys.stderr, flush=True)
eligible = [(idx, mem) for idx, mem in free if mem >= 44000]
if not eligible:
    raise SystemExit(1)
idx, _ = max(eligible, key=lambda item: item[1])
print(idx)
PY
)"
    rc=$?
    if [ "$rc" -eq 0 ] && [ -n "$best" ]; then
      log "DPO GPU selected: $best" >&2
      echo "$best"
      return 0
    fi
    log "DPO GPU memory not ready; need one GPU >= 44000MiB; sleep 120s" >&2
    sleep 120
  done
}

write_config() {
  local run_name="$1"
  local work_dir="$2"
  local model_path="$3"
  local config_path="$4"
  local dpo_gpu="$5"
  local train_max_length="$6"
  CURRENT_STAGE="write_config_${run_name}"
  write_state
  "$PYTHON" - "$run_name" "$work_dir" "$model_path" "$config_path" "$dpo_gpu" "$train_max_length" <<'PY'
import json
import sys
from pathlib import Path

run_name, work_dir, model_path, config_path, dpo_gpu, train_max_length = sys.argv[1:7]
base = Path("configs/strict_final_leader_soft_train64.json")
cfg = json.loads(base.read_text())
cfg["run_name"] = run_name
cfg["resume_stages"] = True
cfg.setdefault("paths", {})["work_dir"] = work_dir
cfg.setdefault("model", {})["model_path"] = model_path
cfg["model"]["input_planner_adapter_path"] = None
cfg["model"]["input_coder_adapter_path"] = None
cfg["model"]["cuda_visible_devices"] = str(dpo_gpu)
cfg["model"]["device"] = "cuda:0"
cfg.setdefault("sampling", {})["sample_limit"] = 64
cfg["sampling"]["sample_max_rounds"] = 2
cfg["sampling"]["planner_temperatures"] = [0.2, 0.5, 0.8]
cfg["sampling"]["follower_temperatures"] = [0.2, 0.5, 0.8]
cfg["sampling"]["coder_temperature"] = 0.2
cfg["sampling"]["top_p"] = 0.95
cfg.setdefault("training", {}).update({
    "leader_train_steps": 20,
    "follower_train_steps": 40,
    "train_max_samples": 256,
    "train_max_length": int(train_max_length),
    "learning_rate": 1e-6,
    "follower_learning_rate": 1e-6,
    "leader_learning_rate": 1e-6,
    "beta": 0.1,
    "follower_beta": 0.1,
    "leader_beta": 0.1,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "follower_lora_rank": 8,
    "follower_lora_alpha": 16,
    "follower_lora_dropout": 0.05,
    "leader_lora_rank": 8,
    "leader_lora_alpha": 16,
    "leader_lora_dropout": 0.05,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "leader_batch_size": 1,
    "leader_gradient_accumulation_steps": 8,
    "follower_batch_size": 1,
    "follower_gradient_accumulation_steps": 8,
})
cfg.setdefault("evaluation", {})["eval_split"] = "test"
cfg["evaluation"]["eval_limit"] = 17
cfg["evaluation"]["eval_max_rounds"] = 2
cfg["evaluation"]["prompt_profile"] = "legacy"
cfg["evaluation"]["best_so_far"] = True
cfg["evaluation"]["coder_adapter_start_round"] = 2
cfg.setdefault("experiment", {})["max_rounds"] = 2
cfg.setdefault("parallel_sampling", {})["enabled"] = True
cfg["parallel_sampling"]["num_shards"] = 4
cfg["parallel_sampling"]["cuda_visible_devices"] = ["0", "1", "2", "3"]
out = Path(config_path)
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
json.loads(out.read_text())
print(out)
PY
}

cleanup_incomplete_adapters() {
  local work_dir="$1"
  for role in follower leader; do
    local adapter_dir="$work_dir/adapters/$role"
    if [ -d "$adapter_dir" ] && [ ! -f "$adapter_dir/adapter_model.safetensors" ]; then
      log "remove incomplete adapter dir: $adapter_dir"
      rm -rf "$adapter_dir"
    fi
  done
}

run_pipeline_once() {
  local config_path="$1"
  CURRENT_STAGE="strict_alternating_smoke"
  write_state
  log "START strict-alternating-smoke config=$config_path"
  "$PYTHON" scripts/main.py strict-alternating-smoke --config "$config_path"
  local rc=$?
  log "strict-alternating-smoke config=$config_path exited rc=$rc"
  return "$rc"
}

summary_for_model() {
  local label="$1"
  local summary_dir="$2"
  local run_name="$3"
  local work_dir="$4"
  local status="$5"
  local config_path="$6"
  CURRENT_STAGE="summary_${label}"
  write_state
  "$PYTHON" - "$label" "$summary_dir" "$run_name" "$work_dir" "$status" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

label, summary_dir, run_name, work_dir, status, config_path = sys.argv[1:7]
summary_dir = Path(summary_dir)
work_dir = Path(work_dir)
summary_dir.mkdir(parents=True, exist_ok=True)
eval_summaries = sorted((work_dir / "eval").glob("*summary.json"))
manifest = work_dir / "manifest.json"
result = {
    "label": label,
    "status": status,
    "run_name": run_name,
    "work_dir": str(work_dir),
    "config_path": config_path,
    "manifest_path": str(manifest),
    "manifest_exists": manifest.exists(),
    "leader_adapter_exists": (work_dir / "adapters/leader/adapter_model.safetensors").exists(),
    "follower_adapter_exists": (work_dir / "adapters/follower/adapter_model.safetensors").exists(),
}
if eval_summaries:
    result["eval_summary_path"] = str(eval_summaries[-1])
    try:
        result["eval_summary"] = json.loads(eval_summaries[-1].read_text())
    except Exception as exc:
        result["eval_summary_error"] = repr(exc)
if manifest.exists():
    try:
        result["manifest"] = json.loads(manifest.read_text())
    except Exception as exc:
        result["manifest_error"] = repr(exc)
json_name = f"{label}_coagent_conservative_train64_summary.json"
md_name = f"{label}_coagent_conservative_train64_summary.md"
(summary_dir / json_name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
lines = [f"# {label} Conservative CoAgent Train64 on A6000", "", f"- status: {status}"]
if "eval_summary" in result:
    s = result["eval_summary"]
    final_rate = s.get("final_pass_rate_binary")
    if final_rate is None and s.get("num_tasks"):
        final_rate = s.get("final_passed", 0) / s["num_tasks"]
    lines += [
        f"- eval_summary: `{result['eval_summary_path']}`",
        f"- final_passed: {s.get('final_passed')}",
        f"- num_tasks: {s.get('num_tasks')}",
        f"- final_pass_rate_binary: {final_rate}",
        f"- avg_assert_pass_rate_final: {s.get('avg_assert_pass_rate_final')}",
        f"- avg_rounds_used: {s.get('avg_rounds_used')}",
        f"- avg_total_tokens: {s.get('avg_total_tokens')}",
    ]
else:
    lines.append("- No eval summary detected.")
(summary_dir / md_name).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

run_model() {
  local label="$1"
  local run_name="$2"
  local model_path="$3"
  local remote_model_path="$4"
  local summary_dir="$5"
  local work_dir="$6"
  local config_path="$7"

  CURRENT_MODEL="$label"
  CURRENT_STAGE="model_start"
  write_state
  log "============================================================"
  log "START model=$label"

  transfer_model_if_needed "$label" "$model_path" "$remote_model_path" || {
    summary_for_model "$label" "$summary_dir" "$run_name" "$work_dir" "model_transfer_failed" "$config_path"
    return 1
  }

  wait_sampling_gpus
  dpo_gpu="$(choose_dpo_gpu)"
  write_config "$run_name" "$work_dir" "$model_path" "$config_path" "$dpo_gpu" 1024
  cleanup_incomplete_adapters "$work_dir"

  run_pipeline_once "$config_path"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    if grep -R "OutOfMemoryError\\|CUDA out of memory" "$work_dir/logs" >/dev/null 2>&1; then
      log "$label detected OOM; fallback train_max_length=768 and retry failed stages"
      cleanup_incomplete_adapters "$work_dir"
      wait_sampling_gpus
      dpo_gpu="$(choose_dpo_gpu)"
      write_config "$run_name" "$work_dir" "$model_path" "$config_path" "$dpo_gpu" 768
      run_pipeline_once "$config_path"
      rc=$?
    fi
  fi

  if [ "$rc" -eq 0 ]; then
    summary_for_model "$label" "$summary_dir" "$run_name" "$work_dir" "completed" "$config_path"
  else
    summary_for_model "$label" "$summary_dir" "$run_name" "$work_dir" "failed_rc_${rc}" "$config_path"
  fi
  log "END model=$label rc=$rc"
  return "$rc"
}

main() {
  CURRENT_MODEL="none"
  CURRENT_STAGE="starting"
  write_state
  log "============================================================"
  log "general_models_conservative_coagent_train64_a6000"
  log "host=$(hostname)"
  log "started_at=$(date '+%F %T')"
  log "params=sample_limit=64 max_rounds=2 eval_limit=17 train_max_samples=256 train_max_length=1024 fallback=768 lr=1e-6 rank=8 alpha=16 batch=1 grad_accum=8"

  run_model \
    "gemma2_9b" \
    "general_base_gemma2_9b_it_coagent_conservative_train64" \
    "/workspace/models/gemma-2-9b-it" \
    "/workspace/models/gemma-2-9b-it" \
    "$ROOT/outputs/gemma2_9b_coagent_conservative_train64_a6000" \
    "$ROOT/outputs/gemma2_9b_coagent_conservative_train64_a6000/general_base_gemma2_9b_it_coagent_conservative_train64" \
    "$ROOT/configs/general_base_gemma2_9b_it_coagent_conservative_train64.json"
  gemma_rc=$?

  run_model \
    "mistral_7b" \
    "general_base_mistral_7b_instruct_v03_coagent_conservative_train64" \
    "/workspace/models/Mistral-7B-Instruct-v0.3" \
    "/workspace/models/Mistral-7B-Instruct-v0.3" \
    "$ROOT/outputs/mistral_7b_coagent_conservative_train64_a6000" \
    "$ROOT/outputs/mistral_7b_coagent_conservative_train64_a6000/general_base_mistral_7b_instruct_v03_coagent_conservative_train64" \
    "$ROOT/configs/general_base_mistral_7b_instruct_v03_coagent_conservative_train64.json"
  mistral_rc=$?

  log "finished_at=$(date '+%F %T') gemma_rc=$gemma_rc mistral_rc=$mistral_rc"
  if [ "$gemma_rc" -eq 0 ] && [ "$mistral_rc" -eq 0 ]; then
    exit 0
  fi
  exit 1
}

main

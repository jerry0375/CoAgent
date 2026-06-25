#!/usr/bin/env bash
set -u
cd /workspace/multi_agent/stackelberg_codepo
mkdir -p outputs/verification_logs outputs/gemma2_9b_coagent_conservative_train64_a6000
LOG="outputs/verification_logs/gemma2_9b_coagent_conservative_train64_a6000_master.log"
PIDFILE="outputs/verification_logs/gemma2_9b_coagent_conservative_train64_a6000.pid"
HEARTBEAT="outputs/verification_logs/gemma2_9b_coagent_conservative_train64_a6000.heartbeat"
CONFIG="configs/general_base_gemma2_9b_it_coagent_conservative_train64.json"
MODEL_DIR="/workspace/models/gemma-2-9b-it"
TRANSFER_DIR="/workspace/models/gemma-2-9b-it.transfer"
SUMMARY_DIR="outputs/gemma2_9b_coagent_conservative_train64_a6000"
RUN_DIR="$SUMMARY_DIR/general_base_gemma2_9b_it_coagent_conservative_train64"
REMOTE_MODEL="root@219.216.65.85:/workspace/models/gemma-2-9b-it"
REMOTE_PORT="32791"
REMOTE_PASSWORD="${REMOTE_PASSWORD:-}"

exec >> "$LOG" 2>&1
echo $$ > "$PIDFILE"

log() {
  echo "[$(date '+%F %T')] $*"
}

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      echo "stage=${CURRENT_STAGE:-starting}"
      if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader || true
      fi
    } > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HB_PID=$!
cleanup() {
  kill "$HB_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "============================================================"
log "gemma2_9b_coagent_conservative_train64_a6000"
log "host=$(hostname)"
log "started_at=$(date '+%F %T')"
log "params=follower_steps=40 leader_steps=20 lr=1e-6 rank=8 sample_limit=64 max_rounds=2 parallel_sampling=4xA6000"

validate_model_dir() {
  local d="$1"
  [ -d "$d" ] || return 1
  [ -f "$d/config.json" ] || return 1
  [ -f "$d/model.safetensors.index.json" ] || return 1
  [ -f "$d/tokenizer.json" ] || [ -f "$d/tokenizer.model" ] || return 1
  for i in 1 2 3 4; do
    [ -f "$d/model-0000${i}-of-00004.safetensors" ] || return 1
  done
  if find "$d" -name '*.incomplete' -o -name '*.partial' | grep -q .; then
    return 1
  fi
  return 0
}

scp_with_password() {
  /opt/conda/bin/python - <<'PY'
import os, pty, select, sys, time, errno
password = os.environ.get('REMOTE_PASSWORD')
if not password:
    print('REMOTE_PASSWORD is required for password-based transfer', file=sys.stderr)
    raise SystemExit(2)
cmd = [
    'scp', '-r', '-P', os.environ.get('REMOTE_PORT', '32791'),
    '-o', 'StrictHostKeyChecking=no',
    '-o', 'UserKnownHostsFile=/root/.ssh/known_hosts',
    os.environ['REMOTE_MODEL'], os.environ['TRANSFER_DIR'],
]
print('server_to_server_cmd=' + ' '.join(cmd), flush=True)
pid, fd = pty.fork()
if pid == 0:
    os.execvp(cmd[0], cmd)
password_sent = False
status = None
while True:
    try:
        r, _, _ = select.select([fd], [], [], 1.0)
    except OSError:
        r = []
    if fd in r:
        try:
            data = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno == errno.EIO:
                data = b''
            else:
                raise
        if not data:
            pass
        else:
            text = data.decode(errors='replace')
            sys.stdout.write(text)
            sys.stdout.flush()
            low = text.lower()
            if 'are you sure' in low:
                os.write(fd, b'yes\n')
            if 'password:' in low and not password_sent:
                os.write(fd, (password + '\n').encode())
                password_sent = True
    done_pid, done_status = os.waitpid(pid, os.WNOHANG)
    if done_pid == pid:
        if os.WIFEXITED(done_status):
            status = os.WEXITSTATUS(done_status)
        else:
            status = 128
        break
    time.sleep(0.05)
sys.exit(status if status is not None else 1)
PY
}

transfer_model_if_needed() {
  CURRENT_STAGE="model_check"
  if validate_model_dir "$MODEL_DIR"; then
    log "model already valid: $MODEL_DIR"
    return 0
  fi
  log "model missing or incomplete on A6000; will pull directly from A800 to A6000"
  mkdir -p /workspace/models
  for attempt in 1 2 3; do
    CURRENT_STAGE="model_transfer_attempt_${attempt}"
    log "transfer attempt $attempt/3: A800 -> A6000, not via local machine"
    rm -rf "$TRANSFER_DIR"
    export REMOTE_PASSWORD REMOTE_PORT REMOTE_MODEL TRANSFER_DIR
    scp_with_password
    rc=$?
    log "scp attempt $attempt exited rc=$rc"
    if validate_model_dir "$TRANSFER_DIR"; then
      rm -rf "$MODEL_DIR.incomplete"
      if [ -d "$MODEL_DIR" ]; then
        mv "$MODEL_DIR" "$MODEL_DIR.incomplete.$(date +%s)"
      fi
      mv "$TRANSFER_DIR" "$MODEL_DIR"
      log "model transfer validated and installed: $MODEL_DIR"
      return 0
    fi
    log "transfer attempt $attempt did not produce a complete model; retry after 60s"
    sleep 60
  done
  log "ERROR: failed to transfer complete Gemma model after 3 attempts"
  return 1
}

wait_gpus() {
  CURRENT_STAGE="waiting_for_gpu_memory"
  local min_free=30000
  while true; do
    /opt/conda/bin/python - <<'PY'
import subprocess, sys
min_free = 30000
out = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
free = []
for line in out.strip().splitlines():
    idx, mem = [x.strip() for x in line.split(',')]
    free.append((idx, int(mem)))
print('gpu_free_mib=' + ','.join(f'{i}:{m}' for i,m in free))
sys.exit(0 if len(free) >= 4 and all(m >= min_free for _,m in free[:4]) else 1)
PY
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "GPU memory ready"
      return 0
    fi
    log "GPU memory not ready for 4-way sampling; sleep 120s"
    sleep 120
  done
}

write_config() {
  CURRENT_STAGE="write_config"
  /opt/conda/bin/python - <<'PY'
import json
from pathlib import Path
base = Path('configs/strict_final_leader_soft_train64.json')
out = Path('configs/general_base_gemma2_9b_it_coagent_conservative_train64.json')
c = json.loads(base.read_text())
c['run_name'] = 'general_base_gemma2_9b_it_coagent_conservative_train64'
c.setdefault('paths', {})['work_dir'] = 'outputs/gemma2_9b_coagent_conservative_train64_a6000/general_base_gemma2_9b_it_coagent_conservative_train64'
c.setdefault('model', {})['model_path'] = '/workspace/models/gemma-2-9b-it'
c['model']['input_planner_adapter_path'] = None
c['model']['input_coder_adapter_path'] = None
c['model']['cuda_visible_devices'] = '0'
c['model']['device'] = 'cuda:0'
c.setdefault('sampling', {})['sample_limit'] = 64
c['sampling']['sample_max_rounds'] = 2
c['sampling']['planner_temperatures'] = [0.2, 0.5, 0.8]
c['sampling']['follower_temperatures'] = [0.2, 0.5, 0.8]
c['sampling']['coder_temperature'] = 0.2
c['sampling']['top_p'] = 0.95
c.setdefault('training', {}).update({
    'leader_train_steps': 20,
    'follower_train_steps': 40,
    'learning_rate': 1e-6,
    'follower_learning_rate': 1e-6,
    'leader_learning_rate': 1e-6,
    'lora_rank': 8,
    'lora_alpha': 16,
    'follower_lora_rank': 8,
    'follower_lora_alpha': 16,
    'leader_lora_rank': 8,
    'leader_lora_alpha': 16,
    'batch_size': 1,
    'gradient_accumulation_steps': 8,
    'leader_batch_size': 1,
    'leader_gradient_accumulation_steps': 8,
    'follower_batch_size': 1,
    'follower_gradient_accumulation_steps': 8,
})
c.setdefault('evaluation', {})['eval_split'] = 'test'
c['evaluation']['eval_limit'] = 17
c['evaluation']['eval_max_rounds'] = 2
c['evaluation']['prompt_profile'] = 'legacy'
c['evaluation']['best_so_far'] = True
c['evaluation']['coder_adapter_start_round'] = 2
c.setdefault('experiment', {})['max_rounds'] = 2
c.setdefault('parallel_sampling', {})['enabled'] = True
c['parallel_sampling']['num_shards'] = 4
c['parallel_sampling']['cuda_visible_devices'] = ['0', '1', '2', '3']
out.write_text(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
print(out)
PY
  /opt/conda/bin/python -m json.tool "$CONFIG" >/dev/null
  log "config written and JSON-validated: $CONFIG"
}

write_summary() {
  CURRENT_STAGE="write_summary"
  /opt/conda/bin/python - <<'PY'
import json
from pathlib import Path
summary_dir = Path('outputs/gemma2_9b_coagent_conservative_train64_a6000')
run_dir = summary_dir / 'general_base_gemma2_9b_it_coagent_conservative_train64'
summary_dir.mkdir(parents=True, exist_ok=True)
manifest = run_dir / 'manifest.json'
metrics = {'run_dir': str(run_dir), 'manifest_exists': manifest.exists()}
if manifest.exists():
    try:
        m = json.loads(manifest.read_text())
        metrics['manifest'] = m
    except Exception as exc:
        metrics['manifest_error'] = repr(exc)
for p in sorted(run_dir.rglob('*summary*.json')):
    try:
        obj=json.loads(p.read_text())
    except Exception:
        continue
    if isinstance(obj, dict) and ('final_passed' in obj or 'num_tasks' in obj or 'avg_assert_pass_rate_final' in obj):
        metrics['detected_summary_path']=str(p)
        metrics['detected_summary']=obj
summary_json = summary_dir / 'gemma2_9b_coagent_conservative_train64_summary.json'
summary_md = summary_dir / 'gemma2_9b_coagent_conservative_train64_summary.md'
summary_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + '\n')
lines=['# Gemma2 9B Conservative CoAgent Train64 on A6000', '']
if 'detected_summary' in metrics:
    s=metrics['detected_summary']
    final=s.get('final_pass_rate_binary')
    if final is None and s.get('num_tasks'):
        final=s.get('final_passed',0)/s.get('num_tasks')
    lines += [
      f"- summary: `{metrics.get('detected_summary_path')}`",
      f"- final_passed: {s.get('final_passed')}",
      f"- num_tasks: {s.get('num_tasks')}",
      f"- final_pass_rate_binary: {final}",
      f"- avg_assert_pass_rate_final: {s.get('avg_assert_pass_rate_final')}",
      f"- avg_rounds_used: {s.get('avg_rounds_used')}",
    ]
else:
    lines.append('- No eval summary detected yet. Check manifest/log for failure stage.')
summary_md.write_text('\n'.join(lines)+'\n')
print(summary_md.read_text())
PY
}

main() {
  transfer_model_if_needed || exit 2
  wait_gpus
  write_config || exit 3
  CURRENT_STAGE="strict_alternating_smoke"
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  log "START strict-alternating-smoke"
  /opt/conda/bin/python scripts/main.py strict-alternating-smoke --config "$CONFIG"
  rc=$?
  log "strict-alternating-smoke exited rc=$rc"
  write_summary || true
  exit "$rc"
}

main

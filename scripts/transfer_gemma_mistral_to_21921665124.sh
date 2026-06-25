#!/usr/bin/env bash
set -u
ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1
LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/transfer_gemma_mistral_to_21921665124_master.log"
PIDFILE="$LOG_DIR/transfer_gemma_mistral_to_21921665124.pid"
HEARTBEAT="$LOG_DIR/transfer_gemma_mistral_to_21921665124.heartbeat"
STATEFILE="$LOG_DIR/transfer_gemma_mistral_to_21921665124.state"
LOCKFILE="$LOG_DIR/transfer_gemma_mistral_to_21921665124.lock"
mkdir -p "$LOG_DIR"
exec >> "$LOG" 2>&1

if [ -f "$LOCKFILE" ]; then
  old_pid="$(cat "$LOCKFILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] existing transfer runner active pid=$old_pid"
    exit 0
  fi
fi
echo $$ > "$LOCKFILE"
echo $$ > "$PIDFILE"

DEST_HOST="219.216.65.124"
DEST_PORT="32791"
DEST_USER="root"
DEST_PASSWORD="${DEST_PASSWORD:-}"
DEST_BASE="/workspace/models"
CURRENT_MODEL="none"
CURRENT_STAGE="starting"
PYTHON="/opt/conda/bin/python"

write_state() { { echo "model=$CURRENT_MODEL"; echo "stage=$CURRENT_STAGE"; } > "$STATEFILE"; }
log() { echo "[$(date '+%F %T')] $*"; }

heartbeat_loop() {
  while true; do
    {
      echo "time=$(date '+%F %T')"
      echo "pid=$$"
      if [ -f "$STATEFILE" ]; then cat "$STATEFILE"; else echo "model=$CURRENT_MODEL"; echo "stage=$CURRENT_STAGE"; fi
      echo "dest=${DEST_USER}@${DEST_HOST}:${DEST_PORT}:${DEST_BASE}"
      for d in /workspace/models/gemma-2-9b-it /workspace/models/Mistral-7B-Instruct-v0.3; do
        [ -d "$d" ] && du -sh "$d" 2>/dev/null || true
      done
    } > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HB=$!
cleanup() { kill "$HB" >/dev/null 2>&1 || true; rm -f "$LOCKFILE"; }
trap cleanup EXIT
write_state

pty_run() {
  DEST_PASSWORD="$DEST_PASSWORD" "$PYTHON" - "$@" <<'PY'
import errno, os, pty, select, sys, time
password = os.environ.get('DEST_PASSWORD')
if not password:
    print('DEST_PASSWORD is required for password-based transfer', file=sys.stderr)
    raise SystemExit(2)
cmd = sys.argv[1:]
print('cmd=' + ' '.join(cmd), flush=True)
pid, fd = pty.fork()
if pid == 0:
    os.execvp(cmd[0], cmd)
status = 1
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
        if data:
            text = data.decode(errors='replace')
            sys.stdout.write(text)
            sys.stdout.flush()
            low = text.lower()
            if 'are you sure' in low:
                os.write(fd, b'yes\n')
            if 'password:' in low:
                os.write(fd, (password + '\n').encode())
    done_pid, done_status = os.waitpid(pid, os.WNOHANG)
    if done_pid == pid:
        status = os.WEXITSTATUS(done_status) if os.WIFEXITED(done_status) else 128
        break
    time.sleep(0.05)
raise SystemExit(status)
PY
}

ssh_remote() {
  local remote_cmd="$1"
  pty_run ssh -p "$DEST_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/root/.ssh/known_hosts \
    "${DEST_USER}@${DEST_HOST}" "$remote_cmd"
}

scp_to_remote() {
  local src="$1"
  local dest="$2"
  pty_run scp -r -P "$DEST_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/root/.ssh/known_hosts \
    "$src" "${DEST_USER}@${DEST_HOST}:$dest"
}

validate_local_model() {
  local label="$1"
  local d="$2"
  [ -d "$d" ] && [ -f "$d/config.json" ] || return 1
  [ -f "$d/tokenizer.json" ] || [ -f "$d/tokenizer.model" ] || return 1
  if [ "$label" = "gemma2_9b" ]; then
    for i in 1 2 3 4; do [ -f "$d/model-0000${i}-of-00004.safetensors" ] || return 1; done
  else
    { [ -f "$d/consolidated.safetensors" ] || { [ -f "$d/model-00001-of-00003.safetensors" ] && [ -f "$d/model-00002-of-00003.safetensors" ] && [ -f "$d/model-00003-of-00003.safetensors" ]; }; } || return 1
  fi
  return 0
}

remote_validate_shell_cmd() {
  local label="$1"
  local d="$2"
  if [ "$label" = "gemma2_9b" ]; then
    printf "test -d %q && test -f %q/config.json && { test -f %q/tokenizer.json || test -f %q/tokenizer.model; } && test -f %q/model-00001-of-00004.safetensors && test -f %q/model-00002-of-00004.safetensors && test -f %q/model-00003-of-00004.safetensors && test -f %q/model-00004-of-00004.safetensors && du -sh %q" "$d" "$d" "$d" "$d" "$d" "$d" "$d" "$d" "$d"
  else
    printf "test -d %q && test -f %q/config.json && { test -f %q/tokenizer.json || test -f %q/tokenizer.model; } && { test -f %q/consolidated.safetensors || { test -f %q/model-00001-of-00003.safetensors && test -f %q/model-00002-of-00003.safetensors && test -f %q/model-00003-of-00003.safetensors; }; } && du -sh %q" "$d" "$d" "$d" "$d" "$d" "$d" "$d" "$d" "$d"
  fi
}

transfer_one() {
  local label="$1"
  local src="$2"
  local dest="$DEST_BASE/$(basename "$src")"
  local transfer="${dest}.transfer"
  CURRENT_MODEL="$label"; CURRENT_STAGE="validate_local"; write_state
  log "START $label src=$src dest=${DEST_HOST}:$dest"
  if ! validate_local_model "$label" "$src"; then log "ERROR local model invalid: $src"; return 2; fi

  CURRENT_STAGE="remote_prepare_${label}"; write_state
  ssh_remote "mkdir -p '$DEST_BASE' && rm -rf '$transfer'"
  local rc=$?
  if [ "$rc" -ne 0 ]; then log "ERROR remote prepare failed rc=$rc"; return "$rc"; fi

  for attempt in 1 2 3; do
    CURRENT_STAGE="scp_${label}_attempt_${attempt}"; write_state
    log "$label transfer attempt $attempt/3"
    ssh_remote "rm -rf '$transfer'"
    scp_to_remote "$src" "$transfer"
    rc=$?
    log "$label scp attempt $attempt rc=$rc"
    if [ "$rc" -ne 0 ]; then sleep 60; continue; fi

    CURRENT_STAGE="remote_validate_${label}_attempt_${attempt}"; write_state
    ssh_remote "$(remote_validate_shell_cmd "$label" "$transfer")"
    rc=$?
    log "$label remote validate attempt $attempt rc=$rc"
    if [ "$rc" -eq 0 ]; then
      CURRENT_STAGE="remote_install_${label}"; write_state
      ssh_remote "rm -rf '${dest}.old' && if [ -d '$dest' ]; then mv '$dest' '${dest}.old'; fi && mv '$transfer' '$dest' && du -sh '$dest'"
      rc=$?
      log "$label remote install rc=$rc"
      return "$rc"
    fi
    sleep 60
  done
  log "ERROR $label transfer failed after 3 attempts"
  return 1
}

main() {
  log "============================================================"
  log "transfer_gemma_mistral_to_21921665124_shell_validate"
  log "host=$(hostname) dest=${DEST_USER}@${DEST_HOST}:${DEST_PORT}"
  transfer_one "gemma2_9b" "/workspace/models/gemma-2-9b-it"
  gemma_rc=$?
  transfer_one "mistral_7b" "/workspace/models/Mistral-7B-Instruct-v0.3"
  mistral_rc=$?
  CURRENT_STAGE="done"; write_state
  log "finished gemma_rc=$gemma_rc mistral_rc=$mistral_rc"
  if [ "$gemma_rc" -eq 0 ] && [ "$mistral_rc" -eq 0 ]; then exit 0; fi
  exit 1
}
main

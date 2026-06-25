#!/usr/bin/env bash
set -u
ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT" || exit 1
LOG_DIR="$ROOT/outputs/verification_logs"
LOG="$LOG_DIR/qwen32_coagent_4bit_train64_a800_master.log"
PIDFILE="$LOG_DIR/qwen32_coagent_4bit_train64_a800.pid"
HEARTBEAT="$LOG_DIR/qwen32_coagent_4bit_train64_a800.heartbeat"
CONFIG="$ROOT/configs/general_base_qwen25_coder_32b_coagent_4bit_conservative_train64_a800.json"
WORK_DIR="$ROOT/outputs/qwen25_coder_32b_coagent_4bit_conservative_train64_a800/general_base_qwen25_coder_32b_coagent_4bit_conservative_train64"
PYTHON="/opt/conda/bin/python"
mkdir -p "$LOG_DIR"
echo $$ > "$PIDFILE"
exec >> "$LOG" 2>&1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CURRENT_STAGE="starting"
log(){ echo "[$(date '+%F %T')] $*"; }
heartbeat_loop(){
  while true; do
    {
      echo "time=$(date '+%F %T')"; echo "pid=$$"; echo "stage=$CURRENT_STAGE"; echo "config=$CONFIG"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader,nounits || true
      find "$WORK_DIR" -maxdepth 3 -name '*summary.json' -o -name 'adapter_model.safetensors' 2>/dev/null | sort | tail -40 || true
    } > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop & HB=$!
cleanup(){ kill "$HB" >/dev/null 2>&1 || true; }
trap cleanup EXIT
wait_gpu(){
  CURRENT_STAGE="waiting_gpu"
  while true; do
    free=$($PYTHON - <<'PY'
import subprocess
out=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
print([int(line.split(',')[1].strip()) for line in out.splitlines() if line.split(',')[0].strip()=='0'][0])
PY
)
    log "gpu0_free_mib=$free required=70000"
    [ "$free" -ge 70000 ] && return 0
    sleep 120
  done
}
write_summary(){
  local status="$1"
  CURRENT_STAGE="summary"
  $PYTHON - "$status" <<'PY'
import json, sys
from pathlib import Path
status=sys.argv[1]
work=Path('outputs/qwen25_coder_32b_coagent_4bit_conservative_train64_a800/general_base_qwen25_coder_32b_coagent_4bit_conservative_train64')
evals=sorted((work/'eval').glob('*summary.json'))
data={
 'status': status,
 'config': 'configs/general_base_qwen25_coder_32b_coagent_4bit_conservative_train64_a800.json',
 'load_in_4bit': True,
 'follower_adapter_exists': (work/'adapters/follower/adapter_model.safetensors').exists(),
 'leader_adapter_exists': (work/'adapters/leader/adapter_model.safetensors').exists(),
 'eval_summary_path': str(evals[-1]) if evals else None,
}
if evals:
    data['eval_summary']=json.loads(evals[-1].read_text())
out=Path('outputs/qwen25_coder_32b_coagent_4bit_conservative_train64_a800/qwen32_coagent_4bit_train64_summary.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(data, ensure_ascii=False, indent=2), flush=True)
PY
}
main(){
  log "============================================================"
  log "START qwen32 coagent 4bit train64"
  wait_gpu
  CURRENT_STAGE="strict_alternating_smoke"
  $PYTHON scripts/main.py strict-alternating-smoke --config "$CONFIG"
  rc=$?
  log "strict_alternating_smoke returncode=$rc"
  if [ "$rc" -eq 0 ]; then write_summary completed; else write_summary failed_rc_${rc}; fi
  exit "$rc"
}
main "$@"

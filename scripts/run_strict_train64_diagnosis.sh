#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/multi_agent/stackelberg_codepo"
cd "$ROOT"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TAG="strict_train64_diagnosis"
SOURCE_RUN="strict_guarded_train64_round2_iter0"
EVAL_GPU="${EVAL_GPU:-0}"
WAIT_FREE_MIB="${WAIT_FREE_MIB:-16000}"
LOG_DIR="$ROOT/outputs/verification_logs"
OUT_DIR="$ROOT/outputs/${TAG}"
MASTER_LOG="$LOG_DIR/${TAG}_master.log"
LOCK_FILE="$LOG_DIR/${TAG}.lock"
HEARTBEAT="$LOG_DIR/${TAG}.heartbeat"

mkdir -p "$LOG_DIR" "$OUT_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another ${TAG} runner is already active" >> "$MASTER_LOG"
  exit 1
fi

heartbeat_loop() {
  while true; do
    date '+%Y-%m-%d %H:%M:%S' > "$HEARTBEAT"
    sleep 60
  done
}
heartbeat_loop &
HEARTBEAT_PID=$!

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$MASTER_LOG"
}

on_exit() {
  local rc=$?
  kill "$HEARTBEAT_PID" 2>/dev/null || true
  if [[ "$rc" -ne 0 ]]; then
    log "FAILED rc=${rc}; inspect ${MASTER_LOG} and outputs/${TAG}"
  fi
  exit "$rc"
}
trap on_exit EXIT

wait_for_memory() {
  local reason="$1"
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$EVAL_GPU" | head -n 1 | tr -d ' ')"
    if [[ -n "$free_mib" && "$free_mib" -ge "$WAIT_FREE_MIB" ]]; then
      log "GPU${EVAL_GPU} has enough memory for ${reason}: free=${free_mib} MiB"
      return
    fi
    log "wait GPU${EVAL_GPU} memory for ${reason}: free=${free_mib:-unknown} MiB, need>=${WAIT_FREE_MIB}"
    sleep 300
  done
}

write_configs() {
  /opt/conda/bin/python - <<'PY'
import copy
import json
from pathlib import Path

source_run = "strict_guarded_train64_round2_iter0"
source_dir = Path("outputs") / source_run
source_manifest = source_dir / "manifest.json"
leader_adapter = source_dir / "adapters" / "leader" / "adapter_model.safetensors"
follower_adapter = source_dir / "adapters" / "follower" / "adapter_model.safetensors"
missing = [str(path) for path in [source_manifest, leader_adapter, follower_adapter] if not path.exists()]
if missing:
    raise FileNotFoundError("Missing required train64 artifacts: " + ", ".join(missing))

base_path = Path("configs/strict_guarded_train64_round2_iter0.json")
if base_path.exists():
    base = json.loads(base_path.read_text())
else:
    base = json.loads(Path("configs/ablation_iter0_train131_round2_passonly_repaironly_step50_core.json").read_text())

paths = copy.deepcopy(base.get("paths", {}))
paths.update({
    "full_run_dir": str(source_dir),
    "phase1_dir": paths.get("phase1_dir", "../humaneval_phase1"),
    "task_file": paths.get("task_file", "../humaneval_phase1/data/processed/humaneval/test.jsonl"),
    "train_file": paths.get("train_file", "../humaneval_phase1/data/processed/humaneval/train.jsonl"),
    "valid_file": paths.get("valid_file", "../humaneval_phase1/data/processed/humaneval/valid.jsonl"),
})

model = copy.deepcopy(base.get("model", {}))
model["model_path"] = model.get("model_path", "/workspace/models/Qwen2.5-Coder-1.5B-Instruct")
model["cuda_visible_devices"] = __import__("os").environ.get("EVAL_GPU", "0")
model["device"] = "cuda:0"

common = {
    "paths": paths,
    "model": model,
    "variants": [
        {"name": "prompt_only", "planner_adapter": None, "coder_adapter": None},
        {"name": "leader_only", "planner_adapter": "leader", "coder_adapter": None},
        {"name": "follower_repair_only", "planner_adapter": None, "coder_adapter": "follower"},
        {"name": "full_repair_only", "planner_adapter": "leader", "coder_adapter": "follower"},
    ],
    "cost": copy.deepcopy(base.get("cost", {})),
    "incentive": copy.deepcopy(base.get("incentive", {})),
}

for rounds in [2, 4]:
    cfg = copy.deepcopy(common)
    cfg["run_name"] = f"strict_train64_ablation_round{rounds}"
    cfg["paths"]["output_dir"] = f"outputs/strict_train64_diagnosis/round{rounds}"
    cfg["evaluation"] = {
        "split": "test",
        "limit": 17,
        "max_rounds": rounds,
        "temperature": 0.0,
        "top_p": 1.0,
        "prompt_profile": "legacy",
        "best_so_far": True,
        "coder_adapter_start_round": 2,
    }
    out = Path("configs") / f"strict_train64_ablation_round{rounds}.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    json.loads(out.read_text())
    print(out)
PY
}

write_summary() {
  /opt/conda/bin/python - <<'PY'
import csv
import json
from pathlib import Path

out_dir = Path("outputs/strict_train64_diagnosis")
source_manifest = json.loads(Path("outputs/strict_guarded_train64_round2_iter0/manifest.json").read_text())
leader_clean_wpo = source_manifest.get("counts", {}).get("leader_clean_wpo")
follower_wpo = source_manifest.get("counts", {}).get("follower_wpo")

fieldnames = [
    "setting",
    "variant",
    "max_rounds",
    "final_passed",
    "num_tasks",
    "final_pass_rate_binary",
    "avg_assert_pass_rate_final",
    "avg_rounds_used",
    "avg_total_tokens",
    "avg_total_cost",
    "avg_leader_utility",
    "syntax_error_count",
    "syntax_error_rate",
    "indentation_error_count",
    "leader_clean_wpo",
    "follower_wpo",
]

def error_counts(eval_jsonl: Path) -> tuple[int, int, int]:
    total = 0
    syntax = 0
    indentation = 0
    with eval_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            item = json.loads(line)
            errors = []
            for turn in item.get("turns", []):
                feedback = turn.get("feedback") if isinstance(turn, dict) else {}
                if isinstance(feedback, dict):
                    errors.append(str(feedback.get("error_type") or ""))
            if any("SyntaxError" in value for value in errors):
                syntax += 1
            if any("IndentationError" in value for value in errors):
                indentation += 1
    return total, syntax, indentation

rows = []
for rounds in [2, 4]:
    report_path = out_dir / f"round{rounds}" / "ablation_summary.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing ablation report: {report_path}")
    report = json.loads(report_path.read_text())
    for variant in report.get("variants", []):
        summary = variant.get("summary") or {}
        eval_jsonl = out_dir / f"round{rounds}" / "eval" / f"{variant['name']}.jsonl"
        total, syntax, indentation = error_counts(eval_jsonl)
        if int(summary.get("num_tasks", 0)) != 17:
            raise RuntimeError(f"{variant['name']} round{rounds} num_tasks={summary.get('num_tasks')} expected 17")
        rows.append({
            "setting": f"round{rounds}",
            "variant": variant["name"],
            "max_rounds": rounds,
            "final_passed": summary.get("final_passed"),
            "num_tasks": summary.get("num_tasks"),
            "final_pass_rate_binary": summary.get("final_pass_rate_binary"),
            "avg_assert_pass_rate_final": summary.get("avg_assert_pass_rate_final"),
            "avg_rounds_used": summary.get("avg_rounds_used"),
            "avg_total_tokens": summary.get("avg_total_tokens"),
            "avg_total_cost": summary.get("avg_total_cost"),
            "avg_leader_utility": summary.get("avg_leader_utility"),
            "syntax_error_count": syntax,
            "syntax_error_rate": syntax / total if total else 0.0,
            "indentation_error_count": indentation,
            "leader_clean_wpo": leader_clean_wpo,
            "follower_wpo": follower_wpo,
        })

csv_path = out_dir / "diagnosis_summary.csv"
json_path = out_dir / "diagnosis_summary.json"
md_path = out_dir / "diagnosis_summary.md"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps({"rows": rows, "source_manifest": str(Path("outputs/strict_guarded_train64_round2_iter0/manifest.json"))}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

lines = []
lines.append("| " + " | ".join(fieldnames) + " |")
lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
for row in rows:
    lines.append("| " + " | ".join(fmt(row[name]) for name in fieldnames) + " |")
md = "\n".join(lines) + "\n"
md_path.write_text(md, encoding="utf-8")
print(md, flush=True)
print(f"summary_csv={csv_path}", flush=True)
print(f"summary_json={json_path}", flush=True)
print(f"summary_md={md_path}", flush=True)
PY
}

{
  echo "============================================================"
  echo "$TAG"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "root=$ROOT"
  git rev-parse --short HEAD 2>/dev/null || true
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
} >> "$MASTER_LOG"

write_configs >> "$MASTER_LOG" 2>&1

wait_for_memory "round2 ablation"
log "START round2 ablation"
PYTHONPATH="$ROOT/src" /opt/conda/bin/python scripts/main.py run-ablation --config configs/strict_train64_ablation_round2.json >> "$MASTER_LOG" 2>&1
log "END round2 ablation"

wait_for_memory "round4 ablation"
log "START round4 ablation"
PYTHONPATH="$ROOT/src" /opt/conda/bin/python scripts/main.py run-ablation --config configs/strict_train64_ablation_round4.json >> "$MASTER_LOG" 2>&1
log "END round4 ablation"

log "START diagnosis summary"
write_summary >> "$MASTER_LOG" 2>&1
log "END diagnosis summary"

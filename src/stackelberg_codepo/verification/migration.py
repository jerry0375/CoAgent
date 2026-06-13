from __future__ import annotations

import argparse
import json
from pathlib import Path
import py_compile
import re
from typing import Any

from stackelberg_codepo.alternating import run_full_algorithm_smoke
from stackelberg_codepo.io import load_jsonl, write_json

OLD_DEP_PATTERNS = [
    "rq1" + "_experiment",
    "run" + "_alternating" + "_iteration.py",
    "train" + "_weighted" + "_dpo" + "_smoke.py",
    "build" + "_leader" + "_preferences" + "_demo.py",
    "build" + "_follower" + "_preferences" + "_from" + "_trajectories.py",
    "demo" + "_role" + "_humaneval.py",
]
TEXT_SUFFIXES = {".py", ".json", ".md", ".toml", ".txt", ".yml", ".yaml"}
PLANNER_OVERREACH = [
    re.compile(r"```", re.IGNORECASE),
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\(", re.IGNORECASE),
    re.compile(r"\bassert\s+", re.IGNORECASE),
    re.compile(r"\b(?:here(?:'s| is)?|below is)\s+(?:the\s+)?(?:implementation|code|function)\b", re.IGNORECASE),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _adapter_exists(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def _ok(checks: list[dict[str, Any]], name: str, passed: bool, details: Any = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "details": details})


def _has_old_dep(text: str) -> list[str]:
    return [pattern for pattern in OLD_DEP_PATTERNS if pattern in text]


def check_static(project_root: Path, checks: list[dict[str, Any]]) -> None:
    offenders = []
    for base in [project_root / "src", project_root / "scripts", project_root / "configs", project_root / "README.md"]:
        paths = [base] if base.is_file() else list(base.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            rel = str(path.relative_to(project_root))
            text = path.read_text(encoding="utf-8", errors="replace")
            hits = _has_old_dep(text)
            if hits:
                offenders.append({"path": rel, "patterns": hits})
    _ok(checks, "no_old_demo_dependency_strings", not offenders, offenders)

    compile_errors = []
    for path in (project_root / "src").rglob("*.py"):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors.append({"path": str(path.relative_to(project_root)), "error": str(exc)})
    try:
        py_compile.compile(str(project_root / "scripts" / "main.py"), doraise=True)
    except py_compile.PyCompileError as exc:
        compile_errors.append({"path": "scripts/main.py", "error": str(exc)})
    _ok(checks, "python_sources_compile", not compile_errors, compile_errors)

    _ok(
        checks,
        "full_algorithm_binding_is_native",
        run_full_algorithm_smoke.__module__ == "stackelberg_codepo.alternating.native_pipeline",
        run_full_algorithm_smoke.__module__,
    )


def check_full_algorithm(project_root: Path, checks: list[dict[str, Any]]) -> None:
    out = project_root / "outputs" / "full_algorithm_smoke"
    manifest_path = out / "manifest.json"
    report_path = out / "new_project_report.json"
    _ok(checks, "full_manifest_exists", manifest_path.exists(), str(manifest_path))
    _ok(checks, "full_report_exists", report_path.exists(), str(report_path))
    if not manifest_path.exists():
        return
    manifest = _read_json(manifest_path)
    _ok(checks, "full_backend_native", manifest.get("backend") == "native_stackelberg_codepo", manifest.get("backend"))

    stages = manifest.get("stages", [])
    _ok(checks, "full_has_7_stages", len(stages) == 7, [s.get("name") for s in stages])
    stage_failures = []
    for stage in stages:
        command = " ".join(str(x) for x in stage.get("command", []))
        if stage.get("returncode") != 0 or "-m stackelberg_codepo." not in command or _has_old_dep(command):
            stage_failures.append({"name": stage.get("name"), "returncode": stage.get("returncode"), "command": command})
    _ok(checks, "full_stages_native_and_successful", not stage_failures, stage_failures)

    counts = manifest.get("counts", {})
    required_counts = ["trajectories", "trajectory_preferences", "leader_round_wpo", "leader_clean_wpo", "follower_wpo"]
    bad_counts = {k: counts.get(k) for k in required_counts if not isinstance(counts.get(k), int) or counts.get(k, 0) <= 0}
    _ok(checks, "full_required_counts_positive", not bad_counts, counts)

    paths = {k: Path(v) for k, v in manifest.get("paths", {}).items() if isinstance(v, str)}
    missing_paths = [k for k, p in paths.items() if k not in {"work_dir", "eval_dir"} and not p.exists()]
    _ok(checks, "full_manifest_paths_exist", not missing_paths, missing_paths)

    _ok(checks, "full_leader_adapter_exists", _adapter_exists(paths.get("leader_adapter", Path("/missing"))), str(paths.get("leader_adapter")))
    _ok(checks, "full_follower_adapter_exists", _adapter_exists(paths.get("follower_adapter", Path("/missing"))), str(paths.get("follower_adapter")))

    eval_summary = manifest.get("summaries", {}).get("eval")
    _ok(checks, "full_eval_summary_present", isinstance(eval_summary, dict) and eval_summary.get("num_tasks", 0) > 0, eval_summary)

    # Leader preferences: chosen utility must exceed rejected utility.
    leader_pref_path = paths.get("trajectory_preferences")
    leader_errors = []
    if leader_pref_path and leader_pref_path.exists():
        for i, row in enumerate(load_jsonl(leader_pref_path), start=1):
            if float(row.get("chosen_utility", 0)) <= float(row.get("rejected_utility", 0)):
                leader_errors.append({"line": i, "task_id": row.get("task_id")})
    _ok(checks, "full_leader_preferences_utility_direction", not leader_errors, leader_errors[:10])

    # Clean leader WPO: chosen planner output should not overreach into code.
    clean_wpo_path = paths.get("leader_clean_wpo")
    overreach = []
    if clean_wpo_path and clean_wpo_path.exists():
        for i, row in enumerate(load_jsonl(clean_wpo_path), start=1):
            chosen = row.get("chosen", {}).get("value", "") if isinstance(row.get("chosen"), dict) else str(row.get("chosen", ""))
            hits = [p.pattern for p in PLANNER_OVERREACH if p.search(chosen)]
            if hits:
                overreach.append({"line": i, "hits": hits})
    _ok(checks, "full_clean_leader_wpo_no_chosen_overreach", not overreach, overreach[:10])

    # Follower preference state consistency.
    follower_dir = out / "follower"
    candidate_files = sorted(follower_dir.glob("*_candidates.jsonl"))
    pref_files = sorted(follower_dir.glob("*_preferences.jsonl"))
    consistency_errors = []
    if candidate_files and pref_files:
        candidates = {row["candidate_id"]: row for row in load_jsonl(candidate_files[-1])}
        for i, pair in enumerate(load_jsonl(pref_files[-1]), start=1):
            c = candidates.get(pair.get("chosen_candidate_id"))
            r = candidates.get(pair.get("rejected_candidate_id"))
            if not c or not r:
                consistency_errors.append({"line": i, "error": "missing_candidate_ref"})
                continue
            for field in ["state_id", "state_text", "planner_instruction", "incentive_rule", "previous_best_pass_rate"]:
                if c.get(field) != r.get(field):
                    consistency_errors.append({"line": i, "error": "state_mismatch", "field": field})
            if pair.get("chosen_code") == pair.get("rejected_code"):
                consistency_errors.append({"line": i, "error": "identical_code"})
            if float(pair.get("chosen_utility", 0)) <= float(pair.get("rejected_utility", 0)):
                consistency_errors.append({"line": i, "error": "bad_utility_direction"})
    else:
        consistency_errors.append({"error": "missing_follower_candidate_or_preference_file"})
    _ok(checks, "full_follower_pairs_state_consistent", not consistency_errors, consistency_errors[:20])


def check_real_and_tiny(project_root: Path, checks: list[dict[str, Any]]) -> None:
    real_manifest_path = project_root / "outputs" / "real_trajectories_smoke" / "manifest.json"
    tiny_manifest_path = project_root / "outputs" / "tiny_smoke" / "manifest.json"
    _ok(checks, "real_manifest_exists", real_manifest_path.exists(), str(real_manifest_path))
    if real_manifest_path.exists():
        real = _read_json(real_manifest_path)
        src = real.get("source", {}).get("trajectories", "")
        _ok(checks, "real_trajectory_source_native", OLD_DEP_PATTERNS[0] not in src and Path(src).exists(), src)
        counts = real.get("counts", {})
        bad = {k: counts.get(k) for k in ["trajectories", "leader_wpo", "follower_wpo"] if counts.get(k, 0) <= 0}
        _ok(checks, "real_counts_positive", not bad, counts)
        training = real.get("training", {})
        command_errors = []
        for role in ["leader", "follower"]:
            cmd = " ".join(str(x) for x in training.get(role, {}).get("command", []))
            if training.get(role, {}).get("returncode") != 0 or "-m stackelberg_codepo.training.weighted_dpo" not in cmd or _has_old_dep(cmd):
                command_errors.append({"role": role, "command": cmd, "returncode": training.get(role, {}).get("returncode")})
        _ok(checks, "real_training_native_and_successful", not command_errors, command_errors)
        adapters = training.get("adapter_paths", {})
        _ok(checks, "real_leader_adapter_exists", _adapter_exists(Path(adapters.get("leader", "/missing"))), adapters.get("leader"))
        _ok(checks, "real_follower_adapter_exists", _adapter_exists(Path(adapters.get("follower", "/missing"))), adapters.get("follower"))
    _ok(checks, "tiny_manifest_exists", tiny_manifest_path.exists(), str(tiny_manifest_path))
    if tiny_manifest_path.exists():
        tiny = _read_json(tiny_manifest_path)
        counts = tiny.get("counts", {})
        _ok(checks, "tiny_counts_positive", all(counts.get(k, 0) > 0 for k in ["trajectories", "leader_clean_preferences", "follower_preferences"]), counts)


def run_verification(project_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    check_static(project_root, checks)
    check_full_algorithm(project_root, checks)
    check_real_and_tiny(project_root, checks)
    passed = all(check["passed"] for check in checks)
    return {
        "passed": passed,
        "num_checks": len(checks),
        "num_failed": sum(1 for check in checks if not check["passed"]),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify that the Stackelberg CodePO migration is complete and native.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    report = run_verification(project_root)
    output = Path(args.output) if args.output else project_root / "outputs" / "migration_verification_report.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report: {output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stackelberg_codepo.alternating import run_from_trajectories, run_full_algorithm_smoke, run_real_paper_smoke, run_tiny_smoke
from stackelberg_codepo.config import load_config
from stackelberg_codepo.verification.migration import run_verification
from stackelberg_codepo.ablation.runner import run_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stackelberg CodePO project CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    tiny = sub.add_parser("tiny-smoke", help="Run the smallest refactored pipeline verification.")
    tiny.add_argument("--config", default="configs/tiny.json")

    from_traj = sub.add_parser("from-trajectories", help="Build paper preference data from real trajectory artifacts.")
    from_traj.add_argument("--config", default="configs/real_trajectories_smoke.json")

    paper = sub.add_parser("real-paper-smoke", help="Build real preference data and run 1-step weighted DPO smoke training.")
    paper.add_argument("--config", default="configs/real_trajectories_smoke.json")

    full = sub.add_parser("full-algorithm-smoke", help="Run sample -> preference -> train -> eval through the native full algorithm pipeline.")
    full.add_argument("--config", default="configs/full_algorithm_smoke.json")

    verify = sub.add_parser("verify-migration", help="Verify native migration gates and artifact invariants.")
    verify.add_argument("--output", default=None)

    ablation = sub.add_parser("run-ablation", help="Run adapter ablation variants using a completed full run.")
    ablation.add_argument("--config", default="configs/ablation_train40_step20_core.json")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "tiny-smoke":
        cfg = load_config(Path(args.config))
        manifest = run_tiny_smoke(cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "from-trajectories":
        cfg = load_config(Path(args.config))
        manifest = run_from_trajectories(cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "real-paper-smoke":
        cfg = load_config(Path(args.config))
        manifest = run_real_paper_smoke(cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "full-algorithm-smoke":
        cfg = load_config(Path(args.config))
        report = run_full_algorithm_smoke(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.command == "verify-migration":
        project_root = Path(__file__).resolve().parents[2]
        report = run_verification(project_root)
        output = Path(args.output) if args.output else project_root / "outputs" / "migration_verification_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"report: {output}", flush=True)
        return 0 if report["passed"] else 1
    if args.command == "run-ablation":
        cfg = load_config(Path(args.config))
        report = run_ablation(cfg)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")

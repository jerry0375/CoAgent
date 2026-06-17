from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from stackelberg_codepo.config import load_config, resolve_path, table
from stackelberg_codepo.io import write_json


def _adapter_path(full_run_dir: Path, spec: str | None) -> Path | None:
    if spec is None:
        return None
    path = Path(spec)
    if path.is_absolute():
        return path
    if spec in {"leader", "follower"}:
        return full_run_dir / "adapters" / spec
    return full_run_dir / spec


def _adapter_exists(path: Path | None) -> bool:
    if path is None:
        return True
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def run_ablation(cfg: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(cfg["_project_root"]).resolve()
    paths = table(cfg, "paths")
    model = table(cfg, "model")
    evaluation = table(cfg, "evaluation")
    output_dir = resolve_path(cfg, paths.get("output_dir", "outputs/ablation"))
    full_run_dir = resolve_path(cfg, paths["full_run_dir"])
    logs_dir = output_dir / "logs"
    eval_dir = output_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = full_run_dir / "manifest.json"
    if not full_manifest.exists():
        raise FileNotFoundError(f"Full run manifest not found: {full_manifest}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if model.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])

    rows: list[dict[str, Any]] = []
    variant_reports: list[dict[str, Any]] = []
    for variant in cfg.get("variants", []):
        name = str(variant["name"])
        planner_adapter = _adapter_path(full_run_dir, variant.get("planner_adapter"))
        coder_adapter = _adapter_path(full_run_dir, variant.get("coder_adapter"))
        if not _adapter_exists(planner_adapter):
            raise FileNotFoundError(f"Planner adapter missing for {name}: {planner_adapter}")
        if not _adapter_exists(coder_adapter):
            raise FileNotFoundError(f"Coder adapter missing for {name}: {coder_adapter}")

        command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.evaluation.role_eval",
            "--config", str(cfg["_config_path"]),
            "--model-path", str(model["model_path"]),
            "--split", str(evaluation.get("split", "test")),
            "--limit", str(evaluation.get("limit", 17)),
            "--output-dir", str(eval_dir),
            "--output-name", name,
            "--device", str(model.get("device", "cuda:0")),
            "--max-rounds", str(evaluation.get("max_rounds", 1)),
            "--temperature", str(evaluation.get("temperature", 0.0)),
            "--top-p", str(evaluation.get("top_p", 1.0)),
            "--prompt-profile", str(evaluation.get("prompt_profile", "legacy")),
            "--no-resume",
        ]
        if planner_adapter is not None:
            command.extend(["--planner-adapter-path", str(planner_adapter)])
        if coder_adapter is not None:
            command.extend(["--coder-adapter-path", str(coder_adapter)])
        if evaluation.get("best_so_far", False):
            command.append("--best-so-far")
        else:
            command.append("--no-best-so-far")
        if evaluation.get("repair_num_samples") is not None:
            command.extend(["--repair-num-samples", str(evaluation["repair_num_samples"])])
        if evaluation.get("repair_temperature") is not None:
            command.extend(["--repair-temperature", str(evaluation["repair_temperature"])])
        if evaluation.get("coder_adapter_start_round") is not None:
            command.extend(["--coder-adapter-start-round", str(evaluation["coder_adapter_start_round"])])

        log_path = logs_dir / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=project_root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        summary_path = eval_dir / f"{name}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        variant_report = {
            "name": name,
            "returncode": proc.returncode,
            "command": command,
            "log_path": str(log_path),
            "summary_path": str(summary_path),
            "planner_adapter": str(planner_adapter) if planner_adapter else None,
            "coder_adapter": str(coder_adapter) if coder_adapter else None,
            "summary": summary,
        }
        variant_reports.append(variant_report)
        if proc.returncode != 0:
            raise RuntimeError(f"Ablation variant failed: {name}; see {log_path}")
        if not summary:
            raise RuntimeError(f"Ablation summary missing for {name}: {summary_path}")
        rows.append({
            "variant": name,
            "num_tasks": summary.get("num_tasks"),
            "first_round_passed": summary.get("first_round_passed"),
            "first_round_pass_at_1": summary.get("first_round_pass_at_1"),
            "final_passed": summary.get("final_passed"),
            "final_pass_rate_binary": summary.get("final_pass_rate_binary"),
            "avg_assert_pass_rate_final": summary.get("avg_assert_pass_rate_final"),
            "avg_rounds_used": summary.get("avg_rounds_used"),
            "avg_total_tokens": summary.get("avg_total_tokens"),
            "avg_total_cost": summary.get("avg_total_cost"),
            "avg_leader_utility": summary.get("avg_leader_utility"),
            "planner_adapter": str(planner_adapter) if planner_adapter else "",
            "coder_adapter": str(coder_adapter) if coder_adapter else "",
        })

    csv_path = output_dir / "ablation_summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "run_name": cfg.get("run_name", "ablation"),
        "full_run_dir": str(full_run_dir),
        "full_manifest": str(full_manifest),
        "output_dir": str(output_dir),
        "num_variants": len(variant_reports),
        "rows": rows,
        "variants": variant_reports,
        "paths": {
            "summary_json": str(output_dir / "ablation_summary.json"),
            "summary_csv": str(csv_path),
            "eval_dir": str(eval_dir),
            "logs_dir": str(logs_dir),
        },
    }
    write_json(output_dir / "ablation_summary.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run role adapter ablations using a completed full native run.")
    parser.add_argument("--config", default="configs/ablation_train40_step20_core.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    report = run_ablation(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

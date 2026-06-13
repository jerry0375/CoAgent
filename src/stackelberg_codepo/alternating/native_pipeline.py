from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from stackelberg_codepo.config import resolve_path, table
from stackelberg_codepo.io import write_json


def _append_if_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_many(command: list[str], flag: str, values: list[Any]) -> None:
    command.append(flag)
    command.extend(str(value) for value in values)


def _count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_stage(name: str, command: list[str], cwd: Path, env: dict[str, str], log_dir: Path, manifest: dict[str, Any]) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    stage = {
        "name": name,
        "command": command,
        "log_path": str(log_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "returncode": None,
    }
    manifest.setdefault("stages", []).append(stage)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
    stage["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stage["returncode"] = proc.returncode
    if proc.returncode != 0:
        raise RuntimeError(f"Stage failed: {name}; see {log_path}")


def _require_nonempty(path: Path, label: str) -> None:
    if not _count_jsonl(path):
        raise RuntimeError(f"{label} is empty or missing: {path}")


def run_full_algorithm_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one native alternating-optimization iteration.

    This is the migrated one-iteration alternating optimization entry:
    sample trajectories, build leader/follower WPO data, train both adapters,
    and evaluate the joint planner-coder model. It intentionally calls only
    stackelberg_codepo modules.
    """

    project_root = Path(cfg["_project_root"]).resolve()
    config_path = Path(cfg["_config_path"]).resolve()
    paths = table(cfg, "paths")
    model = table(cfg, "model")
    sampling = table(cfg, "sampling")
    training = table(cfg, "training")
    evaluation = table(cfg, "evaluation")
    leader_cleaning = table(cfg, "leader_cleaning")

    run_name = str(cfg.get("run_name", "full_algorithm_smoke"))
    work_dir = resolve_path(cfg, paths["work_dir"])
    sample_dir = work_dir / "sample"
    leader_round_dir = work_dir / "leader_round"
    leader_clean_dir = work_dir / "leader_clean"
    follower_dir = work_dir / "follower"
    adapters_dir = work_dir / "adapters"
    eval_dir = work_dir / "eval"
    log_dir = work_dir / "logs"
    manifest_path = work_dir / "manifest.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(model.get("cuda_visible_devices", "1"))

    sample_prefix = f"{run_name}_sample"
    trajectories = sample_dir / f"{sample_prefix}_trajectories.jsonl"
    trajectory_preferences = sample_dir / f"{sample_prefix}_preferences.jsonl"

    leader_all_name = f"{run_name}_leader_round_all"
    leader_round_wpo = leader_round_dir / f"{leader_all_name}_wpo.jsonl"
    leader_clean_wpo = leader_clean_dir / f"{run_name}_leader_initial_clean_wpo.jsonl"
    leader_clean_dpo = Path("/workspace/LLaMA-Factory/data") / f"{run_name}_leader_initial_clean_dpo.json"
    leader_dataset_name = f"{run_name}_leader_initial_clean_dpo"

    follower_prefix = f"{run_name}_follower"
    follower_wpo = follower_dir / f"{follower_prefix}_wpo.jsonl"

    leader_adapter = adapters_dir / "leader"
    follower_adapter = adapters_dir / "follower"

    manifest: dict[str, Any] = {
        "iteration_id": run_name,
        "backend": "native_stackelberg_codepo",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "Native migrated one-iteration alternating optimization: sample, preference construction, weighted DPO, joint evaluation.",
            "This path does not call external demo scripts.",
            "The implementation preserves the old validated demo behavior instead of using the simplified skeleton utilities.",
        ],
        "paths": {
            "work_dir": str(work_dir),
            "manifest": str(manifest_path),
            "trajectories": str(trajectories),
            "trajectory_preferences": str(trajectory_preferences),
            "leader_round_wpo": str(leader_round_wpo),
            "leader_clean_wpo": str(leader_clean_wpo),
            "leader_clean_dpo": str(leader_clean_dpo),
            "follower_wpo": str(follower_wpo),
            "leader_adapter": str(leader_adapter),
            "follower_adapter": str(follower_adapter),
            "eval_dir": str(eval_dir),
        },
        "config": str(config_path),
    }

    try:
        sample_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.alternating.leader_sampling",
            "--config", str(config_path),
            "--model-path", str(model["model_path"]),
            "--split", str(sampling.get("sample_split", "train")),
            "--limit", str(sampling.get("sample_limit", 3)),
            "--output-dir", str(sample_dir),
            "--output-prefix", sample_prefix,
            "--device", str(model.get("device", "cuda:0")),
        ]
        _append_if_value(sample_command, "--planner-adapter-path", model.get("input_planner_adapter_path"))
        _append_if_value(sample_command, "--coder-adapter-path", model.get("input_coder_adapter_path"))
        _append_many(sample_command, "--planner-temperatures", sampling.get("planner_temperatures", [0.2, 0.7, 1.0]))
        sample_command.extend([
            "--coder-temperature", str(sampling.get("coder_temperature", 0.2)),
            "--top-p", str(sampling.get("top_p", 0.95)),
            "--seed", str(sampling.get("seed", 42)),
            "--max-rounds", str(sampling.get("sample_max_rounds", 1)),
        ])
        _run_stage("01_sample_trajectories", sample_command, project_root, env, log_dir, manifest)
        _require_nonempty(trajectories, "trajectories")
        _require_nonempty(trajectory_preferences, "trajectory preferences")

        _run_stage(
            "02_convert_leader_round_pairs",
            [
                "/opt/conda/bin/python", "-m", "stackelberg_codepo.preference.leader_round_conversion",
                "--preferences", str(trajectory_preferences),
                "--trajectories", str(trajectories),
                "--output-dir", str(leader_round_dir),
                "--all-name", leader_all_name,
                "--strong-name", f"{run_name}_leader_round_strong",
            ],
            project_root,
            env,
            log_dir,
            manifest,
        )
        _require_nonempty(leader_round_wpo, "leader round WPO")

        clean_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.preference.leader_cleaning_wpo",
            "--input", str(leader_round_wpo),
            "--output-wpo", str(leader_clean_wpo),
            "--summary", str(leader_clean_dir / f"{run_name}_leader_initial_clean_summary.json"),
            "--dpo-output", str(leader_clean_dpo),
            "--dataset-name", leader_dataset_name,
            "--trajectories", str(trajectories),
            "--max-pairs-per-task", str(leader_cleaning.get("max_pairs_per_task", 3)),
            "--min-response-chars", str(leader_cleaning.get("min_response_chars", 80)),
        ]
        _append_if_value(clean_command, "--min-chosen-pass-rate", leader_cleaning.get("min_chosen_pass_rate"))
        _append_if_value(clean_command, "--min-chosen-utility", leader_cleaning.get("min_chosen_utility"))
        if leader_cleaning.get("require_chosen_passed", False):
            clean_command.append("--require-chosen-passed")
        if leader_cleaning.get("drop_rejected_overreach", False):
            clean_command.append("--drop-rejected-overreach")
        _run_stage(
            "03_clean_initial_leader_pairs",
            clean_command,
            project_root,
            env,
            log_dir,
            manifest,
        )
        _require_nonempty(leader_clean_wpo, "clean leader WPO")

        follower_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.alternating.follower_sampling",
            "--config", str(config_path),
            "--model-path", str(model["model_path"]),
            "--trajectories", str(trajectories),
            "--output-dir", str(follower_dir),
            "--output-prefix", follower_prefix,
            "--device", str(model.get("device", "cuda:0")),
            "--limit-states", str(sampling.get("follower_limit_states", 4)),
        ]
        _append_many(follower_command, "--temperatures", sampling.get("follower_temperatures", [0.2, 0.8]))
        follower_command.extend([
            "--top-p", str(sampling.get("top_p", 0.95)),
            "--seed", str(sampling.get("seed", 42)),
        ])
        _run_stage("04_build_follower_pairs", follower_command, project_root, env, log_dir, manifest)
        _require_nonempty(follower_wpo, "follower WPO")

        def train_command(role: str, data: Path, out: Path, steps_key: str) -> list[str]:
            return [
                "/opt/conda/bin/python", "-m", "stackelberg_codepo.training.weighted_dpo",
                "--model-path", str(model["model_path"]),
                "--data", str(data),
                "--output-dir", str(out),
                "--device", str(model.get("device", "cuda:0")),
                "--max-steps", str(training.get(steps_key, 1)),
                "--max-samples", str(training.get("train_max_samples", 8)),
                "--max-length", str(training.get("train_max_length", 1024)),
                "--learning-rate", str(training.get("learning_rate", 5e-6)),
                "--beta", str(training.get("beta", 0.1)),
                "--normalize-logprob",
            ]

        _run_stage("05_train_leader", train_command("leader", leader_clean_wpo, leader_adapter, "leader_train_steps"), project_root, env, log_dir, manifest)
        _run_stage("06_train_follower", train_command("follower", follower_wpo, follower_adapter, "follower_train_steps"), project_root, env, log_dir, manifest)

        _run_stage(
            "07_eval_joint",
            [
                "/opt/conda/bin/python", "-m", "stackelberg_codepo.evaluation.role_eval",
                "--config", str(config_path),
                "--model-path", str(model["model_path"]),
                "--planner-adapter-path", str(leader_adapter),
                "--coder-adapter-path", str(follower_adapter),
                "--split", str(evaluation.get("eval_split", "test")),
                "--limit", str(evaluation.get("eval_limit", 3)),
                "--output-dir", str(eval_dir),
                "--output-name", f"{run_name}_leader_follower_eval",
                "--device", str(model.get("device", "cuda:0")),
                "--max-rounds", str(evaluation.get("eval_max_rounds", 1)),
                "--temperature", "0.0",
                "--top-p", "1.0",
                "--prompt-profile", str(evaluation.get("prompt_profile", "legacy")),
                "--no-resume",
            ],
            project_root,
            env,
            log_dir,
            manifest,
        )
    finally:
        manifest["counts"] = {
            "trajectories": _count_jsonl(trajectories),
            "trajectory_preferences": _count_jsonl(trajectory_preferences),
            "leader_round_wpo": _count_jsonl(leader_round_wpo),
            "leader_clean_wpo": _count_jsonl(leader_clean_wpo),
            "follower_wpo": _count_jsonl(follower_wpo),
        }
        manifest["summaries"] = {
            "sample": _read_json_if_exists(sample_dir / f"{sample_prefix}_summary.json"),
            "leader_clean": _read_json_if_exists(leader_clean_dir / f"{run_name}_leader_initial_clean_summary.json"),
            "follower": _read_json_if_exists(follower_dir / f"{follower_prefix}_summary.json"),
            "eval": _read_json_if_exists(eval_dir / f"{run_name}_leader_follower_eval_summary.json"),
        }
        write_json(manifest_path, manifest)

    report = {
        "run_name": run_name,
        "backend": "native_stackelberg_codepo",
        "returncode": 0,
        "work_dir": str(work_dir),
        "manifest_path": str(manifest_path),
        "counts": manifest.get("counts"),
        "eval": manifest.get("summaries", {}).get("eval"),
        "artifacts": {
            "leader_adapter": str(leader_adapter),
            "follower_adapter": str(follower_adapter),
            "leader_clean_wpo_count": _count_jsonl(leader_clean_wpo),
            "follower_wpo_count": _count_jsonl(follower_wpo),
        },
    }
    write_json(work_dir / "new_project_report.json", report)
    return report

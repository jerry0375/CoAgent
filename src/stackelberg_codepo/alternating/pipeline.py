from __future__ import annotations

from pathlib import Path
import os
import subprocess
from typing import Any

from stackelberg_codepo.config import resolve_path, table
from stackelberg_codepo.execution import run_function_tests
from stackelberg_codepo.io import load_jsonl, write_json, write_jsonl
from stackelberg_codepo.preference import build_follower_preferences, build_leader_preferences, clean_leader_preferences
from stackelberg_codepo.preference.formatting import follower_pair_to_wpo, leader_pair_to_wpo
from stackelberg_codepo.preference.leader import enrich_trajectory
from stackelberg_codepo.schemas import Task, UtilityConfig


def _tasks_by_id(rows: list[dict[str, Any]]) -> dict[str, Task]:
    return {
        row["task_id"]: Task(
            task_id=row["task_id"],
            entry_point=row["entry_point"],
            prompt=row["prompt"],
            tests=row["tests"],
        )
        for row in rows
    }


def run_tiny_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = table(cfg, "paths")
    utility_cfg = UtilityConfig.from_dict(table(cfg, "utility"))
    execution_cfg = table(cfg, "execution")
    output_dir = resolve_path(cfg, paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = _tasks_by_id(load_jsonl(resolve_path(cfg, paths["tasks"])))
    raw_trajectories = load_jsonl(resolve_path(cfg, paths["trajectories"]))

    enriched_trajectories = []
    follower_candidates = []
    for row in raw_trajectories:
        task = tasks[row["task_id"]]
        feedback = run_function_tests(
            row["code"],
            task.entry_point,
            task.tests,
            timeout_seconds=float(execution_cfg.get("timeout_seconds", 2.0)),
        )
        enriched = enrich_trajectory(row, float(feedback["pass_rate"]), utility_cfg)
        enriched["feedback"] = feedback
        enriched_trajectories.append(enriched)
        follower_candidates.append(
            {
                "task_id": row["task_id"],
                "state_id": f"{row['task_id']}_initial",
                "code": row["code"],
                "pass_rate": feedback["pass_rate"],
                "coder_tokens": row.get("coder_tokens", 0),
                "incentive": 0.0,
            }
        )

    leader_pairs = build_leader_preferences(enriched_trajectories, utility_cfg)
    clean_leader_pairs, clean_stats = clean_leader_preferences(leader_pairs)
    follower_pairs = build_follower_preferences(follower_candidates, utility_cfg)

    trajectories_path = output_dir / "trajectories_scored.jsonl"
    leader_path = output_dir / "leader_preferences.jsonl"
    clean_leader_path = output_dir / "leader_clean_preferences.jsonl"
    follower_path = output_dir / "follower_preferences.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(trajectories_path, enriched_trajectories)
    write_jsonl(leader_path, leader_pairs)
    write_jsonl(clean_leader_path, clean_leader_pairs)
    write_jsonl(follower_path, follower_pairs)

    manifest = {
        "run_name": cfg.get("run_name", "tiny_smoke"),
        "paths": {
            "trajectories": str(trajectories_path),
            "leader_preferences": str(leader_path),
            "leader_clean_preferences": str(clean_leader_path),
            "follower_preferences": str(follower_path),
            "manifest": str(manifest_path),
        },
        "counts": {
            "tasks": len(tasks),
            "trajectories": len(enriched_trajectories),
            "leader_preferences": len(leader_pairs),
            "leader_clean_preferences": len(clean_leader_pairs),
            "follower_preferences": len(follower_pairs),
        },
        "leader_cleaning": clean_stats,
        "notes": [
            "Tiny smoke uses fixture trajectories and local Python execution.",
            "It verifies project module boundaries without model loading or DPO training.",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def _limit_trajectories(rows: list[dict[str, Any]], max_tasks: int | None, max_trajectories: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if max_tasks is not None and task_id not in seen_tasks and len(seen_tasks) >= max_tasks:
            continue
        if max_trajectories is not None and len(selected) >= max_trajectories:
            break
        selected.append(row)
        seen_tasks.add(task_id)
    return selected


def _normalize_real_trajectory(row: dict[str, Any], cfg: UtilityConfig) -> dict[str, Any]:
    communication_tokens = int(row.get("communication_tokens", 0))
    context_tokens = int(row.get("context_tokens", 0))
    if not communication_tokens and row.get("turns"):
        communication_tokens = sum(int(turn.get("plan_tokens", 0)) + int(turn.get("code_tokens", 0)) for turn in row["turns"])
    total_tokens = context_tokens + communication_tokens
    normalized = {
        "trajectory_id": row.get("trajectory_id"),
        "task_id": row.get("task_id"),
        "source_task_id": row.get("source_task_id"),
        "entry_point": row.get("entry_point"),
        "rounds": int(row.get("rounds", 1)),
        "plan": row.get("plan") or (row.get("turns") or [{}])[0].get("plan", ""),
        "code": row.get("code") or (row.get("turns") or [{}])[-1].get("code", ""),
        "pass_rate": float(row.get("pass_rate", row.get("metrics", {}).get("pass_rate", 0.0))),
        "planner_tokens": sum(int(turn.get("plan_tokens", 0)) for turn in row.get("turns", [])),
        "coder_tokens": sum(int(turn.get("code_tokens", 0)) for turn in row.get("turns", [])),
        "total_tokens": total_tokens,
        "turns": row.get("turns", []),
        "raw": row,
    }
    enriched = enrich_trajectory(normalized, normalized["pass_rate"], cfg)
    # Preserve richer trajectory utility when it exists because it includes
    # incentive and role-boundary terms from the sampling stage.
    if "leader_utility" in row:
        enriched["leader_utility"] = float(row["leader_utility"])
        enriched["quality"] = float(row.get("quality", enriched.get("quality", 0.0)))
        enriched["total_cost"] = float(row.get("total_cost", enriched.get("total_cost", 0.0)))
        enriched["incentive_total"] = float(row.get("incentive_total", 0.0))
        enriched["role_boundary_penalty_total"] = float(row.get("role_boundary_penalty_total", 0.0))
    return enriched


def _follower_candidates_from_real_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        turns = row.get("turns") or []
        for turn in turns:
            round_id = int(turn.get("round", 1))
            state_text = row.get("state_text", "")
            plan = turn.get("plan", row.get("plan", ""))
            prompt = (
                f"Task state:\n{state_text}\n\n"
                f"Planner guidance:\n{plan}\n\n"
                "Write the complete Python function requested by the task. Return Python code only."
            )
            candidates.append(
                {
                    "task_id": row.get("task_id"),
                    "state_id": f"{row.get('task_id')}_round{round_id}",
                    "trajectory_id": row.get("trajectory_id"),
                    "round": round_id,
                    "prompt": prompt,
                    "code": turn.get("code", row.get("code", "")),
                    "pass_rate": float(turn.get("pass_rate", row.get("pass_rate", 0.0))),
                    "coder_tokens": int(turn.get("code_tokens", 0)),
                    "incentive": float(turn.get("incentive", 0.0)),
                }
            )
    return candidates


def run_from_trajectories(cfg: dict[str, Any]) -> dict[str, Any]:
    paths = table(cfg, "paths")
    limits = table(cfg, "limits")
    utility_cfg = UtilityConfig.from_dict(table(cfg, "utility"))
    output_dir = resolve_path(cfg, paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_path = resolve_path(cfg, paths["trajectories"])
    raw_rows = load_jsonl(trajectory_path)
    rows = _limit_trajectories(
        raw_rows,
        int(limits["max_tasks"]) if limits.get("max_tasks") is not None else None,
        int(limits["max_trajectories"]) if limits.get("max_trajectories") is not None else None,
    )
    trajectories = [_normalize_real_trajectory(row, utility_cfg) for row in rows]
    leader_pairs = build_leader_preferences(trajectories, utility_cfg)
    clean_leader_pairs, clean_stats = clean_leader_preferences(leader_pairs)
    follower_candidates = _follower_candidates_from_real_trajectories(rows)
    follower_pairs = build_follower_preferences(follower_candidates, utility_cfg)
    trajectories_by_id = {str(row.get("trajectory_id")): row for row in trajectories}
    leader_wpo = [leader_pair_to_wpo(pair, trajectories_by_id) for pair in clean_leader_pairs]
    follower_wpo = [follower_pair_to_wpo(pair) for pair in follower_pairs]

    scored_path = output_dir / "trajectories_scored.jsonl"
    leader_path = output_dir / "leader_preferences.jsonl"
    clean_leader_path = output_dir / "leader_clean_preferences.jsonl"
    leader_wpo_path = output_dir / "leader_wpo.jsonl"
    follower_candidates_path = output_dir / "follower_candidates.jsonl"
    follower_path = output_dir / "follower_preferences.jsonl"
    follower_wpo_path = output_dir / "follower_wpo.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(scored_path, trajectories)
    write_jsonl(leader_path, leader_pairs)
    write_jsonl(clean_leader_path, clean_leader_pairs)
    write_jsonl(leader_wpo_path, leader_wpo)
    write_jsonl(follower_candidates_path, follower_candidates)
    write_jsonl(follower_path, follower_pairs)
    write_jsonl(follower_wpo_path, follower_wpo)

    tasks = sorted({str(row.get("task_id")) for row in trajectories})
    manifest = {
        "run_name": cfg.get("run_name", "from_trajectories"),
        "source": {
            "trajectories": str(trajectory_path),
            "num_raw_trajectories": len(raw_rows),
        },
        "paths": {
            "trajectories_scored": str(scored_path),
            "leader_preferences": str(leader_path),
            "leader_clean_preferences": str(clean_leader_path),
            "leader_wpo": str(leader_wpo_path),
            "follower_candidates": str(follower_candidates_path),
            "follower_preferences": str(follower_path),
            "follower_wpo": str(follower_wpo_path),
            "manifest": str(manifest_path),
        },
        "counts": {
            "tasks": len(tasks),
            "trajectories": len(trajectories),
            "leader_preferences": len(leader_pairs),
            "leader_clean_preferences": len(clean_leader_pairs),
            "leader_wpo": len(leader_wpo),
            "follower_candidates": len(follower_candidates),
            "follower_preferences": len(follower_pairs),
            "follower_wpo": len(follower_wpo),
        },
        "leader_cleaning": clean_stats,
        "utility": {
            "leader_margin": utility_cfg.leader_margin,
            "follower_margin": utility_cfg.follower_margin,
            "weight_min": utility_cfg.weight_min,
            "weight_max": utility_cfg.weight_max,
            "weight_power": utility_cfg.weight_power,
        },
        "notes": [
            "This command reuses native HumanEval trajectories produced by the full algorithm smoke run.",
            "Leader utility is preserved from the sampled trajectory when available, so incentive and role-boundary terms are retained.",
            "Follower pairs are reconstructed from trajectory turn code candidates grouped by task and round.",
        ],
    }
    write_json(manifest_path, manifest)
    return manifest


def _run_training_command(command: list[str], log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {
        "command": command,
        "log_path": str(log_path),
        "returncode": proc.returncode,
    }


def run_real_paper_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = run_from_trajectories(cfg)
    training_cfg = table(cfg, "training")
    if not bool(training_cfg.get("enabled", False)):
        manifest["training"] = {"enabled": False}
        write_json(Path(manifest["paths"]["manifest"]), manifest)
        return manifest

    output_dir = resolve_path(cfg, training_cfg.get("output_dir", "outputs/real_trajectories_smoke/adapters"))
    model_path = Path(str(training_cfg["model_path"]))
    project_root = Path(cfg["_project_root"]).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if training_cfg.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(training_cfg["cuda_visible_devices"])

    def command(role: str, data_path: str) -> list[str]:
        return [
            "/opt/conda/bin/python",
            "-m",
            "stackelberg_codepo.training.weighted_dpo",
            "--model-path",
            str(model_path),
            "--data",
            data_path,
            "--output-dir",
            str(output_dir / role),
            "--device",
            str(training_cfg.get("device", "cuda:0")),
            "--max-steps",
            str(training_cfg.get("max_steps", 1)),
            "--max-samples",
            str(training_cfg.get("max_samples", 4)),
            "--max-length",
            str(training_cfg.get("max_length", 1024)),
            "--learning-rate",
            str(training_cfg.get("learning_rate", 5e-6)),
            "--beta",
            str(training_cfg.get("beta", 0.1)),
            "--normalize-logprob",
        ]

    logs_dir = output_dir / "logs"
    leader = _run_training_command(command("leader", manifest["paths"]["leader_wpo"]), logs_dir / "train_leader.log", env)
    if leader["returncode"] != 0:
        manifest["training"] = {"enabled": True, "leader": leader, "failed_stage": "leader"}
        write_json(Path(manifest["paths"]["manifest"]), manifest)
        raise RuntimeError(f"Leader training failed; see {leader['log_path']}")
    follower = _run_training_command(command("follower", manifest["paths"]["follower_wpo"]), logs_dir / "train_follower.log", env)
    if follower["returncode"] != 0:
        manifest["training"] = {"enabled": True, "leader": leader, "follower": follower, "failed_stage": "follower"}
        write_json(Path(manifest["paths"]["manifest"]), manifest)
        raise RuntimeError(f"Follower training failed; see {follower['log_path']}")

    manifest["training"] = {
        "enabled": True,
        "leader": leader,
        "follower": follower,
        "adapter_paths": {
            "leader": str(output_dir / "leader"),
            "follower": str(output_dir / "follower"),
        },
    }
    write_json(Path(manifest["paths"]["manifest"]), manifest)
    return manifest

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from stackelberg_codepo.config import resolve_path, table
from stackelberg_codepo.io import write_json
from stackelberg_codepo.alternating.follower_sampling import apply_global_weights, write_wpo_jsonl


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


def _adapter_exists(path: Path) -> bool:
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


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


def _skip_stage(name: str, command: list[str], log_dir: Path, manifest: dict[str, Any], reason: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    stage = {
        "name": name,
        "command": command,
        "log_path": str(log_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "returncode": 0,
        "skipped": True,
        "skip_reason": reason,
    }
    manifest.setdefault("stages", []).append(stage)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"skipped": True, "reason": reason, "at": stage["finished_at"]}, ensure_ascii=False) + "\n")


def _run_or_skip_stage(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    manifest: dict[str, Any],
    *,
    completed: bool,
    resume: bool,
    reason: str,
) -> None:
    if resume and completed:
        _skip_stage(name, command, log_dir, manifest, reason)
        return
    _run_stage(name, command, cwd, env, log_dir, manifest)


def _merge_jsonl(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        out.write(line)
                        count += 1
    return count


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rewrite_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _trajectory_health(trajectories_path: Path, preferences_path: Path | None = None) -> dict[str, Any]:
    rows = _load_jsonl(trajectories_path)
    total = len(rows)
    passed = 0
    positive_pass = 0
    syntax_errors = 0
    indent_errors = 0
    round_sum = 0.0
    for row in rows:
        if bool(row.get("passed")):
            passed += 1
        if float(row.get("best_pass_rate", row.get("pass_rate", 0.0)) or 0.0) > 0.0:
            positive_pass += 1
        round_sum += float(row.get("rounds", 0.0) or 0.0)
        turns = row.get("turns") if isinstance(row.get("turns"), list) else []
        final_feedback = row.get("feedback") if isinstance(row.get("feedback"), dict) else {}
        error_types = [str(final_feedback.get("error_type") or "")]
        for turn in turns:
            feedback = turn.get("feedback") if isinstance(turn, dict) else {}
            if isinstance(feedback, dict):
                error_types.append(str(feedback.get("error_type") or ""))
        if any("SyntaxError" in item for item in error_types):
            syntax_errors += 1
        if any("IndentationError" in item for item in error_types):
            indent_errors += 1
    preferences = _count_jsonl(preferences_path) if preferences_path is not None else None
    return {
        "trajectories": total,
        "preferences": preferences,
        "passed": passed,
        "positive_pass_rate": positive_pass,
        "passed_rate": passed / total if total else 0.0,
        "positive_pass_rate_fraction": positive_pass / total if total else 0.0,
        "syntax_error_trajectories": syntax_errors,
        "indentation_error_trajectories": indent_errors,
        "syntax_error_rate": syntax_errors / total if total else 0.0,
        "indentation_error_rate": indent_errors / total if total else 0.0,
        "avg_rounds": round_sum / total if total else 0.0,
    }


def _record_health_gate(
    manifest: dict[str, Any],
    *,
    name: str,
    summary: dict[str, Any],
    checks: dict[str, Any],
    enabled: bool,
) -> None:
    failed: list[str] = []
    if enabled:
        for metric, rule in checks.items():
            if rule is None:
                continue
            value = summary.get(metric)
            if value is None:
                failed.append(f"{metric}=missing")
                continue
            if "min" in rule and float(value) < float(rule["min"]):
                failed.append(f"{metric}={value} < {rule['min']}")
            if "max" in rule and float(value) > float(rule["max"]):
                failed.append(f"{metric}={value} > {rule['max']}")
    record = {
        "name": name,
        "enabled": enabled,
        "summary": summary,
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest.setdefault("health_gates", []).append(record)
    if failed:
        raise RuntimeError(f"Health gate failed: {name}: {failed}")


def _rebuild_follower_wpo_with_global_weights(
    candidates_path: Path,
    preferences_path: Path,
    wpo_path: Path,
    *,
    weight_min: float,
    weight_max: float,
    weight_power: float,
    partial_chosen_weight_scale: float,
) -> int:
    pairs = _load_jsonl(preferences_path)
    candidates = {row["candidate_id"]: row for row in _load_jsonl(candidates_path)}
    apply_global_weights(pairs, weight_min, weight_max, weight_power, partial_chosen_weight_scale)
    _rewrite_jsonl(pairs, preferences_path)
    return write_wpo_jsonl(pairs, candidates, wpo_path)


def _run_parallel_sampling_stage(
    name: str,
    base_command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    output_prefix: str,
    merged_outputs: dict[str, Path],
    num_shards: int,
    devices: list[str],
    id_flag: str,
    id_prefix: str,
    resume: bool,
    completed: bool,
    reason: str,
) -> None:
    if resume and completed:
        _skip_stage(name, base_command, log_dir, manifest, reason)
        return
    if num_shards <= 1:
        command = list(base_command)
        command.extend(["--output-dir", str(output_dir), "--output-prefix", output_prefix, id_flag, id_prefix])
        _run_stage(name, command, cwd, env, log_dir, manifest)
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards" / name
    shard_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    stage = {
        "name": name,
        "command": base_command,
        "parallel": True,
        "num_shards": num_shards,
        "devices": devices,
        "started_at": started_at,
        "finished_at": None,
        "returncode": None,
        "shards": [],
    }
    manifest.setdefault("stages", []).append(stage)

    procs: list[tuple[int, subprocess.Popen, Any, Path, Path, Path]] = []
    for shard_index in range(num_shards):
        shard_prefix = f"{output_prefix}_shard{shard_index}"
        shard_output_dir = shard_dir / f"shard{shard_index}"
        shard_output_dir.mkdir(parents=True, exist_ok=True)
        shard_log_path = log_dir / f"{name}_shard{shard_index}.log"
        command = list(base_command)
        command.extend([
            "--output-dir", str(shard_output_dir),
            "--output-prefix", shard_prefix,
            "--num-shards", str(num_shards),
            "--shard-index", str(shard_index),
            id_flag, f"{id_prefix}_s{shard_index}",
        ])
        shard_env = env.copy()
        if devices:
            shard_env["CUDA_VISIBLE_DEVICES"] = str(devices[shard_index % len(devices)])
        log_handle = shard_log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(command, cwd=cwd, env=shard_env, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
        stage["shards"].append({
            "shard_index": shard_index,
            "command": command,
            "log_path": str(shard_log_path),
            "output_dir": str(shard_output_dir),
            "output_prefix": shard_prefix,
            "cuda_visible_devices": shard_env.get("CUDA_VISIBLE_DEVICES"),
        })
        procs.append((shard_index, proc, log_handle, shard_log_path, shard_output_dir, Path(shard_prefix)))

    failed: list[tuple[int, int, Path]] = []
    for shard_index, proc, log_handle, log_path, _, _ in procs:
        returncode = proc.wait()
        log_handle.close()
        stage["shards"][shard_index]["returncode"] = returncode
        if returncode != 0:
            failed.append((shard_index, returncode, log_path))
    if failed:
        stage["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        stage["returncode"] = failed[0][1]
        raise RuntimeError(f"Parallel stage failed: {name}; failed shards: {failed}")

    for suffix, merged_path in merged_outputs.items():
        shard_files = [
            shard_output_dir / f"{shard_prefix}_{suffix}"
            for _, _, _, _, shard_output_dir, shard_prefix in procs
        ]
        _merge_jsonl(shard_files, merged_path)

    summary = {
        "parallel": True,
        "num_shards": num_shards,
        "devices": devices,
        "merged_outputs": {key: str(value) for key, value in merged_outputs.items()},
    }
    (output_dir / f"{output_prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stage["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stage["returncode"] = 0


def _require_nonempty(path: Path, label: str) -> None:
    if not _count_jsonl(path):
        raise RuntimeError(f"{label} is empty or missing: {path}")


def run_strict_alternating_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one paper-faithful alternating iteration.

    Order:
    1. Fix current leader and sample interaction states for follower data.
    2. Train follower/coder on same-state multi-response preferences.
    3. Fix the updated follower and resample complete trajectories for leader data.
    4. Train leader/planner on trajectory-level preferences.
    5. Evaluate the resulting leader-follower pair.
    """

    project_root = Path(cfg["_project_root"]).resolve()
    config_path = Path(cfg["_config_path"]).resolve()
    paths = table(cfg, "paths")
    model = table(cfg, "model")
    sampling = table(cfg, "sampling")
    parallel_sampling = table(cfg, "parallel_sampling")
    training = table(cfg, "training")
    evaluation = table(cfg, "evaluation")
    leader_cleaning = table(cfg, "leader_cleaning")
    follower_preference = table(cfg, "follower_preference")
    preference = table(cfg, "preference")
    incentive = table(cfg, "incentive")
    health_gates = table(cfg, "health_gates")

    run_name = str(cfg.get("run_name", "strict_alternating_smoke"))
    resume_stages = bool(cfg.get("resume_stages", True))
    work_dir = resolve_path(cfg, paths["work_dir"])
    follower_context_dir = work_dir / "follower_context"
    follower_dir = work_dir / "follower"
    leader_sample_dir = work_dir / "leader_sample"
    leader_round_dir = work_dir / "leader_round"
    leader_clean_dir = work_dir / "leader_clean"
    adapters_dir = work_dir / "adapters"
    eval_dir = work_dir / "eval"
    log_dir = work_dir / "logs"
    manifest_path = work_dir / "manifest.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if model.get("cuda_visible_devices") is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(model["cuda_visible_devices"])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    follower_context_prefix = f"{run_name}_follower_context"
    follower_context_trajectories = follower_context_dir / f"{follower_context_prefix}_trajectories.jsonl"
    follower_context_preferences = follower_context_dir / f"{follower_context_prefix}_preferences.jsonl"
    follower_prefix = f"{run_name}_follower"
    follower_wpo = follower_dir / f"{follower_prefix}_wpo.jsonl"
    follower_adapter = adapters_dir / "follower"

    leader_sample_prefix = f"{run_name}_leader_sample"
    leader_trajectories = leader_sample_dir / f"{leader_sample_prefix}_trajectories.jsonl"
    leader_preferences = leader_sample_dir / f"{leader_sample_prefix}_preferences.jsonl"
    leader_all_name = f"{run_name}_leader_round_all"
    leader_round_wpo = leader_round_dir / f"{leader_all_name}_wpo.jsonl"
    leader_clean_wpo = leader_clean_dir / f"{run_name}_leader_initial_clean_wpo.jsonl"
    leader_clean_dpo = Path("/workspace/LLaMA-Factory/data") / f"{run_name}_leader_initial_clean_dpo.json"
    leader_dataset_name = f"{run_name}_leader_initial_clean_dpo"
    leader_adapter = adapters_dir / "leader"

    manifest: dict[str, Any] = {
        "iteration_id": run_name,
        "backend": "native_stackelberg_codepo_strict_alternating",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "Paper-faithful single alternating iteration.",
            "Follower is updated before leader trajectories are resampled.",
            "Leader training uses trajectories generated with the updated follower adapter.",
        ],
        "paths": {
            "work_dir": str(work_dir),
            "manifest": str(manifest_path),
            "follower_context_trajectories": str(follower_context_trajectories),
            "follower_context_preferences": str(follower_context_preferences),
            "follower_wpo": str(follower_wpo),
            "follower_adapter": str(follower_adapter),
            "leader_trajectories": str(leader_trajectories),
            "leader_preferences": str(leader_preferences),
            "leader_round_wpo": str(leader_round_wpo),
            "leader_clean_wpo": str(leader_clean_wpo),
            "leader_adapter": str(leader_adapter),
            "eval_dir": str(eval_dir),
        },
        "config": str(config_path),
    }

    try:
        parallel_enabled = bool(parallel_sampling.get("enabled", False))
        parallel_shards = int(parallel_sampling.get("num_shards", 1)) if parallel_enabled else 1
        parallel_devices = [str(device) for device in parallel_sampling.get("cuda_visible_devices", [])]
        health_enabled = bool(health_gates.get("enabled", False))

        def health_gate(name: str, summary: dict[str, Any], default_checks: dict[str, Any]) -> None:
            spec = health_gates.get(name, {})
            if spec is None:
                spec = {}
            gate_enabled = health_enabled and bool(spec.get("enabled", True))
            checks = dict(default_checks)
            checks.update(spec.get("checks", {}))
            _record_health_gate(
                manifest,
                name=name,
                summary=summary,
                checks=checks,
                enabled=gate_enabled,
            )

        def sample_command(*, coder_adapter_path: str | None) -> list[str]:
            command = [
                "/opt/conda/bin/python", "-m", "stackelberg_codepo.alternating.leader_sampling",
                "--config", str(config_path),
                "--model-path", str(model["model_path"]),
                "--split", str(sampling.get("sample_split", "train")),
                "--limit", str(sampling.get("sample_limit", 3)),
                "--device", str(model.get("device", "cuda:0")),
            ]
            _append_if_value(command, "--planner-adapter-path", model.get("input_planner_adapter_path"))
            _append_if_value(command, "--coder-adapter-path", coder_adapter_path)
            _append_many(command, "--planner-temperatures", sampling.get("planner_temperatures", [0.2, 0.7, 1.0]))
            command.extend([
                "--coder-temperature", str(sampling.get("coder_temperature", 0.2)),
                "--top-p", str(sampling.get("top_p", 0.95)),
                "--seed", str(sampling.get("seed", 42)),
                "--max-rounds", str(sampling.get("sample_max_rounds", 1)),
            ])
            if bool(sampling.get("load_in_4bit", training.get("load_in_4bit", training.get("dpo_load_in_4bit", False)))):
                command.append("--load-in-4bit")
            elif bool(sampling.get("load_in_8bit", training.get("load_in_8bit", training.get("dpo_load_in_8bit", False)))):
                command.append("--load-in-8bit")
            if not bool(incentive.get("enabled", True)):
                command.append("--disable-incentive")
            return command

        context_command = sample_command(coder_adapter_path=model.get("input_coder_adapter_path"))
        _run_parallel_sampling_stage(
            "01_sample_follower_context",
            context_command,
            project_root,
            env,
            log_dir,
            manifest,
            output_dir=follower_context_dir,
            output_prefix=follower_context_prefix,
            merged_outputs={
                "trajectories.jsonl": follower_context_trajectories,
                "preferences.jsonl": follower_context_preferences,
            },
            num_shards=parallel_shards,
            devices=parallel_devices,
            id_flag="--id-prefix",
            id_prefix="fctx",
            completed=bool(_count_jsonl(follower_context_trajectories)) and bool(_count_jsonl(follower_context_preferences)),
            resume=resume_stages,
            reason="follower context trajectories already exist",
        )
        _require_nonempty(follower_context_trajectories, "follower context trajectories")
        health_gate(
            "01_follower_context_sampling",
            _trajectory_health(follower_context_trajectories, follower_context_preferences),
            {
                "trajectories": {"min": 1},
                "syntax_error_rate": {"max": 0.30},
                "positive_pass_rate_fraction": {"min": 0.05},
                "avg_rounds": {"max": float(sampling.get("sample_max_rounds", 1))},
            },
        )

        follower_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.alternating.follower_sampling",
            "--config", str(config_path),
            "--model-path", str(model["model_path"]),
            "--trajectories", str(follower_context_trajectories),
            "--device", str(model.get("device", "cuda:0")),
            "--limit-states", str(sampling.get("follower_limit_states", 4)),
        ]
        _append_many(follower_command, "--temperatures", sampling.get("follower_temperatures", [0.2, 0.8]))
        _append_if_value(follower_command, "--max-round1-states", follower_preference.get("max_round1_states"))
        _append_if_value(follower_command, "--max-repair-states", follower_preference.get("max_repair_states"))
        _append_if_value(follower_command, "--min-chosen-pass-rate", follower_preference.get("min_chosen_pass_rate"))
        _append_if_value(follower_command, "--min-pass-rate-delta", follower_preference.get("min_pass_rate_delta"))
        _append_if_value(follower_command, "--partial-chosen-weight-scale", follower_preference.get("partial_chosen_weight_scale"))
        if follower_preference.get("prefer_repair_states", False):
            follower_command.append("--prefer-repair-states")
        if follower_preference.get("repair_only_states", False):
            follower_command.append("--repair-only-states")
        if follower_preference.get("require_chosen_passed", False):
            follower_command.append("--require-chosen-passed")
        if follower_preference.get("require_repair_improvement", False):
            follower_command.append("--require-repair-improvement")
        follower_command.extend(["--top-p", str(sampling.get("top_p", 0.95)), "--seed", str(sampling.get("seed", 42))])
        if bool(sampling.get("load_in_4bit", training.get("load_in_4bit", training.get("dpo_load_in_4bit", False)))):
            follower_command.append("--load-in-4bit")
        elif bool(sampling.get("load_in_8bit", training.get("load_in_8bit", training.get("dpo_load_in_8bit", False)))):
            follower_command.append("--load-in-8bit")
        _run_parallel_sampling_stage(
            "02_build_follower_pairs",
            follower_command,
            project_root,
            env,
            log_dir,
            manifest,
            output_dir=follower_dir,
            output_prefix=follower_prefix,
            merged_outputs={
                "candidates.jsonl": follower_dir / f"{follower_prefix}_candidates.jsonl",
                "preferences.jsonl": follower_dir / f"{follower_prefix}_preferences.jsonl",
                "wpo.jsonl": follower_wpo,
            },
            num_shards=parallel_shards,
            devices=parallel_devices,
            id_flag="--candidate-id-prefix",
            id_prefix="fcand",
            completed=bool(_count_jsonl(follower_wpo)),
            resume=resume_stages,
            reason="follower WPO already exists",
        )
        _require_nonempty(follower_wpo, "follower WPO")
        if parallel_shards > 1:
            wpo_examples = _rebuild_follower_wpo_with_global_weights(
                follower_dir / f"{follower_prefix}_candidates.jsonl",
                follower_dir / f"{follower_prefix}_preferences.jsonl",
                follower_wpo,
                weight_min=float(preference.get("weight_min", 0.3)),
                weight_max=float(preference.get("weight_max", 2.0)),
                weight_power=float(preference.get("weight_power", 0.65)),
                partial_chosen_weight_scale=float(follower_preference.get("partial_chosen_weight_scale", 1.0)),
            )
            manifest.setdefault("postprocess", []).append({
                "name": "02b_rebuild_follower_wpo_global_weights",
                "wpo_examples": wpo_examples,
                "wpo_path": str(follower_wpo),
            })
        health_gate(
            "02_follower_preference_data",
            {
                "wpo": _count_jsonl(follower_wpo) or 0,
                "preferences": _count_jsonl(follower_dir / f"{follower_prefix}_preferences.jsonl") or 0,
                "candidates": _count_jsonl(follower_dir / f"{follower_prefix}_candidates.jsonl") or 0,
            },
            {
                "wpo": {"min": 4},
                "preferences": {"min": 4},
                "candidates": {"min": 8},
            },
        )

        def train_command(role: str, data: Path, out: Path, steps_key: str) -> list[str]:
            role_prefix = f"{role}_"
            command = [
                "/opt/conda/bin/python", "-m", "stackelberg_codepo.training.weighted_dpo",
                "--model-path", str(model["model_path"]),
                "--data", str(data),
                "--output-dir", str(out),
                "--device", str(model.get("device", "cuda:0")),
                "--max-steps", str(training.get(steps_key, 1)),
                "--max-samples", str(training.get("train_max_samples", 8)),
                "--max-length", str(training.get("train_max_length", 1024)),
                "--learning-rate", str(training.get(f"{role_prefix}learning_rate", training.get("learning_rate", 5e-6))),
                "--beta", str(training.get(f"{role_prefix}beta", training.get("beta", 0.1))),
                "--lora-rank", str(training.get(f"{role_prefix}lora_rank", training.get("lora_rank", 8))),
                "--lora-alpha", str(training.get(f"{role_prefix}lora_alpha", training.get("lora_alpha", 16))),
                "--lora-dropout", str(training.get(f"{role_prefix}lora_dropout", training.get("lora_dropout", 0.05))),
                "--batch-size", str(training.get(f"{role_prefix}batch_size", training.get("batch_size", 1))),
                "--gradient-accumulation-steps", str(training.get(f"{role_prefix}gradient_accumulation_steps", training.get("gradient_accumulation_steps", 1))),
                "--normalize-logprob",
            ]
            if bool(training.get(f"{role_prefix}load_in_4bit", training.get("load_in_4bit", training.get("dpo_load_in_4bit", False)))):
                command.append("--load-in-4bit")
            elif bool(training.get(f"{role_prefix}load_in_8bit", training.get("load_in_8bit", training.get("dpo_load_in_8bit", False)))):
                command.append("--load-in-8bit")
            adapter_key = "input_planner_adapter_path" if role == "leader" else "input_coder_adapter_path"
            _append_if_value(command, "--adapter-path", model.get(adapter_key))
            return command

        follower_train_command = train_command("follower", follower_wpo, follower_adapter, "follower_train_steps")
        _run_or_skip_stage(
            "03_train_follower",
            follower_train_command,
            project_root,
            env,
            log_dir,
            manifest,
            completed=_adapter_exists(follower_adapter),
            resume=resume_stages,
            reason="follower adapter already exists",
        )

        leader_sample_command = sample_command(coder_adapter_path=str(follower_adapter))
        _run_parallel_sampling_stage(
            "04_sample_leader_trajectories_after_follower",
            leader_sample_command,
            project_root,
            env,
            log_dir,
            manifest,
            output_dir=leader_sample_dir,
            output_prefix=leader_sample_prefix,
            merged_outputs={
                "trajectories.jsonl": leader_trajectories,
                "preferences.jsonl": leader_preferences,
            },
            num_shards=parallel_shards,
            devices=parallel_devices,
            id_flag="--id-prefix",
            id_prefix="lead",
            completed=bool(_count_jsonl(leader_trajectories)) and bool(_count_jsonl(leader_preferences)),
            resume=resume_stages,
            reason="leader trajectories after follower update already exist",
        )
        _require_nonempty(leader_trajectories, "leader trajectories")
        _require_nonempty(leader_preferences, "leader trajectory preferences")
        health_gate(
            "04_leader_sampling_after_follower",
            _trajectory_health(leader_trajectories, leader_preferences),
            {
                "trajectories": {"min": 1},
                "preferences": {"min": 2},
                "syntax_error_rate": {"max": 0.30},
                "positive_pass_rate_fraction": {"min": 0.05},
                "avg_rounds": {"max": float(sampling.get("sample_max_rounds", 1))},
            },
        )

        convert_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.preference.leader_round_conversion",
            "--preferences", str(leader_preferences),
            "--trajectories", str(leader_trajectories),
            "--output-dir", str(leader_round_dir),
            "--all-name", leader_all_name,
            "--strong-name", f"{run_name}_leader_round_strong",
        ]
        _run_or_skip_stage(
            "05_convert_leader_round_pairs",
            convert_command,
            project_root,
            env,
            log_dir,
            manifest,
            completed=bool(_count_jsonl(leader_round_wpo)),
            resume=resume_stages,
            reason="leader round WPO already exists",
        )
        _require_nonempty(leader_round_wpo, "leader round WPO")

        clean_command = [
            "/opt/conda/bin/python", "-m", "stackelberg_codepo.preference.leader_cleaning_wpo",
            "--input", str(leader_round_wpo),
            "--output-wpo", str(leader_clean_wpo),
            "--summary", str(leader_clean_dir / f"{run_name}_leader_initial_clean_summary.json"),
            "--dpo-output", str(leader_clean_dpo),
            "--dataset-name", leader_dataset_name,
            "--trajectories", str(leader_trajectories),
            "--max-pairs-per-task", str(leader_cleaning.get("max_pairs_per_task", 3)),
            "--min-response-chars", str(leader_cleaning.get("min_response_chars", 80)),
        ]
        _append_if_value(clean_command, "--min-chosen-pass-rate", leader_cleaning.get("min_chosen_pass_rate"))
        _append_if_value(clean_command, "--min-chosen-utility", leader_cleaning.get("min_chosen_utility"))
        if leader_cleaning.get("require_chosen_passed", False):
            clean_command.append("--require-chosen-passed")
        if leader_cleaning.get("drop_rejected_overreach", False):
            clean_command.append("--drop-rejected-overreach")
        _run_or_skip_stage(
            "06_clean_leader_pairs",
            clean_command,
            project_root,
            env,
            log_dir,
            manifest,
            completed=bool(_count_jsonl(leader_clean_wpo)),
            resume=resume_stages,
            reason="clean leader WPO already exists",
        )
        _require_nonempty(leader_clean_wpo, "clean leader WPO")
        health_gate(
            "06_leader_clean_data",
            {
                "leader_clean_wpo": _count_jsonl(leader_clean_wpo) or 0,
                "leader_round_wpo": _count_jsonl(leader_round_wpo) or 0,
                "leader_preferences": _count_jsonl(leader_preferences) or 0,
            },
            {
                "leader_clean_wpo": {"min": 4},
                "leader_round_wpo": {"min": 4},
                "leader_preferences": {"min": 4},
            },
        )

        leader_train_command = train_command("leader", leader_clean_wpo, leader_adapter, "leader_train_steps")
        _run_or_skip_stage(
            "07_train_leader",
            leader_train_command,
            project_root,
            env,
            log_dir,
            manifest,
            completed=_adapter_exists(leader_adapter),
            resume=resume_stages,
            reason="leader adapter already exists",
        )

        eval_command = [
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
        ]
        if bool(evaluation.get("load_in_4bit", training.get("load_in_4bit", training.get("dpo_load_in_4bit", False)))):
            eval_command.append("--load-in-4bit")
        elif bool(evaluation.get("load_in_8bit", training.get("load_in_8bit", training.get("dpo_load_in_8bit", False)))):
            eval_command.append("--load-in-8bit")
        if evaluation.get("best_so_far", False):
            eval_command.append("--best-so-far")
        else:
            eval_command.append("--no-best-so-far")
        if not bool(incentive.get("enabled", True)):
            eval_command.extend(["--demo-incentive-budget", "0.0", "--demo-max-step-incentive", "0.0"])
        _append_if_value(eval_command, "--repair-num-samples", evaluation.get("repair_num_samples"))
        _append_if_value(eval_command, "--repair-temperature", evaluation.get("repair_temperature"))
        _append_if_value(eval_command, "--coder-adapter-start-round", evaluation.get("coder_adapter_start_round"))
        eval_summary = eval_dir / f"{run_name}_leader_follower_eval_summary.json"
        eval_jsonl = eval_dir / f"{run_name}_leader_follower_eval.jsonl"
        _run_or_skip_stage(
            "08_eval_joint",
            eval_command,
            project_root,
            env,
            log_dir,
            manifest,
            completed=eval_summary.exists() and bool(_count_jsonl(eval_jsonl)),
            resume=resume_stages,
            reason="joint eval summary already exists",
        )
        eval_health = _read_json_if_exists(eval_summary) or {}
        health_gate(
            "08_joint_eval",
            {
                "num_tasks": eval_health.get("num_tasks", 0),
                "final_pass_rate_binary": eval_health.get("final_pass_rate_binary", 0.0),
                "avg_assert_pass_rate_final": eval_health.get("avg_assert_pass_rate_final", 0.0),
                "avg_rounds_used": eval_health.get("avg_rounds_used", 0.0),
            },
            {
                "num_tasks": {"min": 1},
                "avg_assert_pass_rate_final": {"min": 0.10},
            },
        )
    finally:
        manifest["counts"] = {
            "follower_context_trajectories": _count_jsonl(follower_context_trajectories),
            "follower_wpo": _count_jsonl(follower_wpo),
            "leader_trajectories": _count_jsonl(leader_trajectories),
            "leader_preferences": _count_jsonl(leader_preferences),
            "leader_round_wpo": _count_jsonl(leader_round_wpo),
            "leader_clean_wpo": _count_jsonl(leader_clean_wpo),
        }
        manifest["summaries"] = {
            "follower_context": _read_json_if_exists(follower_context_dir / f"{follower_context_prefix}_summary.json"),
            "follower": _read_json_if_exists(follower_dir / f"{follower_prefix}_summary.json"),
            "leader_sample": _read_json_if_exists(leader_sample_dir / f"{leader_sample_prefix}_summary.json"),
            "leader_clean": _read_json_if_exists(leader_clean_dir / f"{run_name}_leader_initial_clean_summary.json"),
            "eval": _read_json_if_exists(eval_dir / f"{run_name}_leader_follower_eval_summary.json"),
        }
        write_json(manifest_path, manifest)

    report = {
        "run_name": run_name,
        "backend": "native_stackelberg_codepo_strict_alternating",
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
    write_json(work_dir / "strict_project_report.json", report)
    return report

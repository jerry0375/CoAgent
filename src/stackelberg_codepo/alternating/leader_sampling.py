#!/usr/bin/env python3
from __future__ import annotations

from stackelberg_codepo.modeling.chat_template import safe_apply_chat_template

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from stackelberg_codepo.config import ExperimentConfig as Config
from stackelberg_codepo.humaneval import HumanEvalExecutor, Task
from stackelberg_codepo.modeling.humaneval_model import extract_code, iter_tasks, load_model_and_tokenizer


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def compact_feedback(feedback_raw: dict[str, Any], max_asserts: int = 6) -> str:
    fields: dict[str, Any] = {
        "passed": feedback_raw.get("passed"),
        "pass_rate": clamp01(float(feedback_raw.get("pass_rate", 0.0))),
        "passed_tests": feedback_raw.get("passed_tests"),
        "total_tests": feedback_raw.get("total_tests"),
        "timeout": feedback_raw.get("timeout"),
        "error_type": feedback_raw.get("error_type"),
        "error_message": feedback_raw.get("error_message"),
    }
    failed = [item for item in feedback_raw.get("assert_results", []) if not item.get("passed")]
    if failed:
        fields["failed_asserts"] = failed[:max_asserts]
    return json.dumps(fields, ensure_ascii=False, indent=2)


def role_boundary_penalty(plan: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if not bool(cfg.get("enabled", True)):
        return {"penalty": 0.0, "violations": {}, "num_violations": 0}
    text = plan or ""
    lower = text.lower()
    violations = {
        "code_block": bool(re.search(r"```", text)),
        "function_def": bool(re.search(r"(?m)^\s*def\s+\w+\s*\(", text)),
        "assert": bool(re.search(r"\bassert\b", text)),
        "test_scaffold": any(token in lower for token in ["test case", "if __name__", "print(", "all tests passed"]),
        "code_intent": bool(re.search(r"\b(?:revised|corrected|final)\s+code\b|here(?:'s| is)\s+the\s+code|below is\s+the\s+code|python function", lower)),
    }
    penalty = 0.0
    if violations["code_block"]:
        penalty += float(cfg.get("lambda_code_block", 0.08))
    if violations["function_def"]:
        penalty += float(cfg.get("lambda_function_def", 0.10))
    if violations["assert"]:
        penalty += float(cfg.get("lambda_assert", 0.06))
    if violations["test_scaffold"]:
        penalty += float(cfg.get("lambda_test_scaffold", 0.06))
    if violations["code_intent"]:
        penalty += float(cfg.get("lambda_code_intent", 0.03))
    penalty = min(float(cfg.get("max_penalty", 0.25)), penalty)
    return {"penalty": penalty, "violations": violations, "num_violations": sum(1 for v in violations.values() if v)}


def build_leader_state_text(task: Task) -> str:
    return "\n".join(
        [
            f"Task ID: {task.task_id}",
            f"Entry point: {task.entry_point}",
            "Task prompt:",
            task.prompt,
        ]
    )


def planner_messages(task: Task, round_index: int, previous_code: str | None, previous_feedback: dict[str, Any] | None) -> list[dict[str, str]]:
    system = (
        "You are the leader/planner in a planner-coder code generation team. "
        "Give concise, actionable guidance. Do not write code. "
        "The coder will implement exactly one Python function."
    )
    if round_index == 1:
        user = (
            "Create an implementation plan for this HumanEval task.\n"
            "Start with exactly one line: Estimated effort tokens: N\n"
            "N is your estimate of total coder output tokens needed to solve the task, between 80 and 800.\n"
            "Then mention the core algorithm and key edge cases. Do not write code.\n\n"
            f"{task.prompt}"
        )
    else:
        user = (
            "The previous code did not fully solve the task. Give a repair plan for the coder.\n"
            "Mention only the likely defect and the next correction. Do not write code.\n\n"
            f"Task:\n{task.prompt}\n\n"
            f"Previous code:\n{previous_code or ''}\n\n"
            f"Execution feedback:\n{compact_feedback(previous_feedback or {})}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def incentive_rule_text(total_budget: float, max_step_incentive: float, best_pass_rate: float) -> str:
    return (
        "Incentive rule: execution improvements are rewarded by incremental pass-rate gain. "
        f"Remaining trajectory incentive budget is about {total_budget:.4f}; "
        f"per-step payment is capped at {max_step_incentive:.4f}; "
        f"current best pass rate is {best_pass_rate:.3f}."
    )


def coder_messages(
    task: Task,
    plan: str,
    round_index: int,
    previous_code: str | None,
    previous_feedback: dict[str, Any] | None,
    incentive_text: str,
) -> list[dict[str, str]]:
    system = (
        "You are the follower/coder in a planner-coder code generation team. "
        "Return only valid Python code. Do not include markdown fences, prose, tests, or examples. "
        f"Define the function {task.entry_point} exactly."
    )
    if round_index == 1:
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Planner guidance:\n{plan}\n\n"
            f"{incentive_text}\n\n"
            "Now write the complete Python function."
        )
    else:
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Previous code:\n{previous_code or ''}\n\n"
            f"Execution feedback:\n{compact_feedback(previous_feedback or {})}\n\n"
            f"Planner repair guidance:\n{plan}\n\n"
            f"{incentive_text}\n\n"
            "Now write a corrected complete Python function."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_p: float, seed: int) -> tuple[str, str, int, float]:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    prompt = safe_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    elapsed = time.perf_counter() - start
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return prompt, generated, int(new_tokens.shape[-1]), elapsed


def parse_effort_tokens(plan: str) -> int | None:
    patterns = [
        r"Estimated\s+effort\s+tokens\s*:\s*(\d+)",
        r"estimated\s+.*?tokens\s*[:=]\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, plan, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def static_task_budget(task: Task, tokenizer, inc_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    prompt_tokens = token_count(tokenizer, task.prompt)
    test_tokens = token_count(tokenizer, task.test)
    estimate_tokens = int(clamp(prompt_tokens + 0.25 * test_tokens, args.difficulty_estimate_min, args.difficulty_estimate_max))
    details = incentive_budget_from_estimate(estimate_tokens, inc_cfg)
    details["static_prompt_tokens"] = float(prompt_tokens)
    details["static_test_tokens"] = float(test_tokens)
    return details


def build_shuffled_budget_map(tasks: list[Task], tokenizer, inc_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, dict[str, float]]:
    source_details = [static_task_budget(task, tokenizer, inc_cfg, args) for task in tasks]
    shuffled_details = [dict(item) for item in source_details]
    rng = random.Random(int(inc_cfg.get("shuffle_seed", args.seed + 7919)))
    rng.shuffle(shuffled_details)
    return {task.task_id: detail for task, detail in zip(tasks, shuffled_details)}


def incentive_budget_from_estimate(estimate_tokens: int, inc_cfg: dict[str, Any]) -> dict[str, float]:
    base = float(inc_cfg.get("total_budget_base", inc_cfg.get("budget_base", 0.03)))
    ref = float(inc_cfg.get("difficulty_token_ref", 256.0))
    scale = max(1.0, float(inc_cfg.get("difficulty_token_scale", 256.0)))
    gamma = float(inc_cfg.get("difficulty_gamma", 0.45))
    max_multiplier = float(inc_cfg.get("max_difficulty_multiplier", 3.0))
    budget_min = float(inc_cfg.get("total_budget_min", 0.0))
    budget_max = float(inc_cfg.get("total_budget_max", inc_cfg.get("max_step_incentive", 0.25)))
    normalized = clamp((float(estimate_tokens) - ref) / scale, -3.0, 3.0)
    multiplier = clamp(math.exp(gamma * normalized), 1.0 / max_multiplier, max_multiplier)
    total_budget = clamp(base * multiplier, budget_min, budget_max)
    return {
        "difficulty_token_estimate": float(estimate_tokens),
        "difficulty_normalized": normalized,
        "difficulty_multiplier": multiplier,
        "total_incentive_budget": total_budget,
    }


def leader_utility(candidate: dict[str, Any], lambda_round: float, lambda_token: float, cost_token_mode: str) -> dict[str, float]:
    # TODO(efficiency): once the executor reports per-test time and memory
    # constraints, Q should count only testcases that are correct within limits.
    # For now Q is clipped pass_rate, so it stays in [0, 1].
    quality = clamp01(candidate["pass_rate"])
    if cost_token_mode == "outputs_only":
        token_cost_base = candidate["communication_tokens"]
    elif cost_token_mode == "context_and_outputs":
        token_cost_base = candidate["context_tokens"] + candidate["communication_tokens"]
    else:
        raise ValueError(f"Unsupported cost token mode: {cost_token_mode}")
    round_cost = lambda_round * candidate["rounds"]
    token_cost = lambda_token * token_cost_base
    total_cost = round_cost + token_cost
    incentive_total = float(candidate.get("incentive_total", 0.0))
    role_penalty = float(candidate.get("role_boundary_penalty_total", 0.0))
    utility = quality - total_cost - incentive_total - role_penalty
    return {
        "quality": quality,
        "cost_token_base": float(token_cost_base),
        "round_cost": round_cost,
        "token_cost": token_cost,
        "total_cost": total_cost,
        "incentive_total": incentive_total,
        "role_boundary_penalty_total": role_penalty,
        "leader_utility": utility,
    }


def make_preference_pairs(
    candidates: list[dict[str, Any]],
    margin: float,
    weight_min: float,
    weight_max: float,
    weight_power: float,
) -> list[dict[str, Any]]:
    raw_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for a, b in itertools.combinations(candidates, 2):
        diff = float(a["leader_utility"]) - float(b["leader_utility"])
        if abs(diff) <= margin:
            continue
        chosen, rejected = (a, b) if diff > 0 else (b, a)
        raw_pairs.append((chosen, rejected, abs(diff)))

    if not raw_pairs:
        return []

    avg_delta = sum(delta for _, _, delta in raw_pairs) / len(raw_pairs)
    pairs = []
    for chosen, rejected, delta in raw_pairs:
        raw_weight = delta / (avg_delta + 1e-8)
        compressed_weight = raw_weight ** weight_power
        weight = max(weight_min, min(weight_max, compressed_weight))
        pairs.append(
            {
                "task_id": chosen["task_id"],
                "source_task_id": chosen["source_task_id"],
                "state_id": chosen["state_id"],
                "chosen_trajectory_id": chosen["trajectory_id"],
                "rejected_trajectory_id": rejected["trajectory_id"],
                "chosen_plan": chosen["planner_training_text"],
                "rejected_plan": rejected["planner_training_text"],
                "chosen_utility": chosen["leader_utility"],
                "rejected_utility": rejected["leader_utility"],
                "utility_delta": delta,
                "raw_weight": raw_weight,
                "weight": weight,
                "chosen_metrics": chosen["metrics"],
                "rejected_metrics": rejected["metrics"],
                "cost_token_mode": chosen["cost_token_mode"],
            }
        )
    return pairs


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def weight_stats(pairs: list[dict[str, Any]], weight_min: float, weight_max: float) -> dict[str, Any]:
    weights = [float(pair["weight"]) for pair in pairs]
    deltas = [float(pair["utility_delta"]) for pair in pairs]
    if not weights:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "clipped_low": 0,
            "clipped_high": 0,
            "delta_mean": 0.0,
        }
    return {
        "count": len(weights),
        "min": min(weights),
        "max": max(weights),
        "mean": statistics.mean(weights),
        "stdev": statistics.pstdev(weights) if len(weights) > 1 else 0.0,
        "p10": percentile(weights, 0.10),
        "p50": percentile(weights, 0.50),
        "p90": percentile(weights, 0.90),
        "clipped_low": sum(1 for w in weights if w <= weight_min + 1e-8),
        "clipped_high": sum(1 for w in weights if w >= weight_max - 1e-8),
        "delta_mean": statistics.mean(deltas),
        "delta_min": min(deltas),
        "delta_max": max(deltas),
    }


def build_planner_training_text(turns: list[dict[str, Any]]) -> str:
    parts = []
    for turn in turns:
        label = "Initial plan" if turn["round"] == 1 else f"Repair plan round {turn['round']}"
        parts.append(f"{label}:\n{turn['plan'].strip()}")
    return "\n\n".join(parts).strip() + "\n"


def run_trajectory(
    planner_model,
    planner_tokenizer,
    coder_model,
    coder_tokenizer,
    executor: HumanEvalExecutor,
    task: Task,
    task_index: int,
    sample_index: int,
    planner_temperature: float,
    args: argparse.Namespace,
    inc_cfg: dict[str, Any],
    shuffled_budget_map: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    previous_code = None
    previous_feedback = None
    best_pass_rate = 0.0
    turns: list[dict[str, Any]] = []
    difficulty_details: dict[str, float] | None = None
    total_budget = 0.0
    max_step_incentive = float(inc_cfg.get("max_step_incentive", 0.08))
    incentive_enabled = bool(inc_cfg.get("enabled", True)) and not args.disable_incentive

    for round_index in range(1, args.max_rounds + 1):
        planner_seed = args.seed + task_index * 1000 + sample_index * 100 + round_index * 10
        coder_seed = planner_seed + 1
        planner_prompt, plan, plan_new_tokens, planner_time = chat_generate(
            planner_model,
            planner_tokenizer,
            planner_messages(task, round_index, previous_code, previous_feedback),
            args.planner_max_new_tokens,
            planner_temperature,
            args.top_p,
            planner_seed,
        )
        planner_prompt_tokens = token_count(planner_tokenizer, planner_prompt)
        plan_tokens = token_count(planner_tokenizer, plan)
        role_boundary = role_boundary_penalty(plan, getattr(args, "role_boundary_cfg", {}))

        if difficulty_details is None:
            raw_estimate = parse_effort_tokens(plan)
            prompt_tokens = token_count(planner_tokenizer, task.prompt)
            test_tokens = token_count(planner_tokenizer, task.test)
            calibrated_floor = int(plan_tokens + 0.25 * prompt_tokens + 0.10 * test_tokens)
            estimate_tokens = int(clamp(max(raw_estimate or 0, calibrated_floor), args.difficulty_estimate_min, args.difficulty_estimate_max))
            policy = str(inc_cfg.get("policy", "exp_token_difficulty"))
            if incentive_enabled and policy == "shuffled_exp_token_difficulty" and shuffled_budget_map is not None:
                difficulty_details = dict(shuffled_budget_map.get(task.task_id) or incentive_budget_from_estimate(estimate_tokens, inc_cfg))
                difficulty_details["shuffled_from_static_budget"] = 1.0
                difficulty_details["leader_raw_effort_tokens"] = float(raw_estimate or 0)
                difficulty_details["calibrated_floor_tokens"] = float(calibrated_floor)
            else:
                difficulty_details = incentive_budget_from_estimate(estimate_tokens, inc_cfg)
                difficulty_details["leader_raw_effort_tokens"] = float(raw_estimate or 0)
                difficulty_details["calibrated_floor_tokens"] = float(calibrated_floor)
            total_budget = float(difficulty_details["total_incentive_budget"]) if incentive_enabled else 0.0

        incentive_text = incentive_rule_text(total_budget, max_step_incentive, best_pass_rate) if incentive_enabled else "Incentive rule: none."
        coder_prompt, generated, code_new_tokens, coder_time = chat_generate(
            coder_model,
            coder_tokenizer,
            coder_messages(task, plan, round_index, previous_code, previous_feedback, incentive_text),
            args.coder_max_new_tokens,
            args.coder_temperature,
            args.top_p,
            coder_seed,
        )
        code, code_type = extract_code(generated, task)
        feedback = executor.execute(task, code, code_type=code_type)
        pass_rate_raw = float(feedback.raw.get("pass_rate", 0.0))
        pass_rate = clamp01(pass_rate_raw)
        delta_best = max(0.0, pass_rate - best_pass_rate)
        step_incentive = min(max_step_incentive, total_budget * delta_best) if incentive_enabled else 0.0
        best_pass_rate = max(best_pass_rate, pass_rate)

        coder_prompt_tokens = token_count(coder_tokenizer, coder_prompt)
        code_tokens = token_count(coder_tokenizer, code)
        turn = {
            "round": round_index,
            "plan": plan,
            "planner_prompt_tokens": planner_prompt_tokens,
            "plan_tokens": plan_tokens,
            "plan_new_tokens": plan_new_tokens,
            "planner_time": planner_time,
            "role_boundary_penalty": role_boundary["penalty"],
            "role_boundary": role_boundary,
            "coder_prompt_tokens": coder_prompt_tokens,
            "code_tokens": code_tokens,
            "code_new_tokens": code_new_tokens,
            "coder_time": coder_time,
            "generated_text": generated,
            "code": code,
            "code_type": code_type,
            "passed": bool(feedback.passed),
            "pass_rate_raw": pass_rate_raw,
            "pass_rate": pass_rate,
            "delta_best_pass_rate": delta_best,
            "incentive": step_incentive,
            "feedback": feedback.raw,
        }
        turns.append(turn)
        previous_code = code
        previous_feedback = feedback.raw
        if feedback.passed and args.stop_on_pass:
            break

    final_turn = turns[-1]
    state_text = build_leader_state_text(task)
    context_tokens = sum(t["planner_prompt_tokens"] + t["coder_prompt_tokens"] for t in turns)
    communication_tokens = sum(t["plan_tokens"] + t["code_tokens"] for t in turns)
    incentive_total = sum(float(t["incentive"]) for t in turns)
    role_boundary_penalty_total = sum(float(t.get("role_boundary_penalty", 0.0)) for t in turns)
    trajectory = {
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "split": task.split,
        "entry_point": task.entry_point,
        "state_id": f"{task.task_id}_leader_multiround_state",
        "state_text": state_text,
        "state_tokens": token_count(planner_tokenizer, state_text),
        "sample_index": sample_index,
        "planner_temperature": planner_temperature,
        "coder_temperature": args.coder_temperature,
        "top_p": args.top_p,
        "seed": args.seed + task_index * 1000 + sample_index * 100,
        "rounds": len(turns),
        "max_rounds": args.max_rounds,
        "stop_on_pass": args.stop_on_pass,
        "turns": turns,
        "plan": turns[0]["plan"],
        "planner_training_text": build_planner_training_text(turns),
        "generated_text": final_turn["generated_text"],
        "code": final_turn["code"],
        "code_type": final_turn["code_type"],
        "passed": bool(final_turn["passed"]),
        "pass_rate_raw": float(final_turn["pass_rate_raw"]),
        "pass_rate": float(final_turn["pass_rate"]),
        "best_pass_rate": best_pass_rate,
        "feedback": final_turn["feedback"],
        "context_tokens": context_tokens,
        "communication_tokens": communication_tokens,
        "cost_token_mode": args.cost_token_mode,
        "incentive_rule": str(inc_cfg.get("policy", "exp_token_difficulty")) if incentive_enabled else "none",
        "incentive_total": incentive_total,
        "role_boundary_penalty_total": role_boundary_penalty_total,
        "incentive_budget": total_budget,
        "difficulty": difficulty_details or {},
    }
    trajectory.update(leader_utility(trajectory, args.lambda_round, args.lambda_token, args.cost_token_mode))
    trajectory["metrics"] = {
        "passed": trajectory["passed"],
        "pass_rate": trajectory["pass_rate"],
        "pass_rate_raw": trajectory["pass_rate_raw"],
        "best_pass_rate": trajectory["best_pass_rate"],
        "quality": trajectory["quality"],
        "rounds": trajectory["rounds"],
        "context_tokens": trajectory["context_tokens"],
        "communication_tokens": trajectory["communication_tokens"],
        "cost_token_base": trajectory["cost_token_base"],
        "total_cost": trajectory["total_cost"],
        "incentive_total": trajectory["incentive_total"],
        "role_boundary_penalty_total": trajectory["role_boundary_penalty_total"],
        "leader_utility": trajectory["leader_utility"],
        "difficulty_token_estimate": trajectory["difficulty"].get("difficulty_token_estimate"),
        "error_type": final_turn["feedback"].get("error_type"),
        "error_message": final_turn["feedback"].get("error_message"),
    }
    return trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leader/planner multi-round trajectory preference data from HumanEval.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_algorithm_smoke.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--planner-adapter-path", default=None, help="Optional PEFT/LoRA adapter for planner generation.")
    parser.add_argument("--coder-adapter-path", default=None, help="Optional PEFT/LoRA adapter for coder generation.")
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--shuffle-tasks", action="store_true", help="Shuffle the selected split before applying --limit.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split selected tasks into this many modulo shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Run only tasks whose selected index belongs to this shard.")
    parser.add_argument("--id-prefix", default="traj", help="Prefix for generated trajectory IDs.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "preference_demo"))
    parser.add_argument("--output-prefix", default="leader_pref_demo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--planner-temperatures", nargs="+", type=float, default=[0.2, 0.6, 0.8, 1.0])
    parser.add_argument("--planner-max-new-tokens", type=int, default=160)
    parser.add_argument("--coder-temperature", type=float, default=0.0)
    parser.add_argument("--coder-max-new-tokens", type=int, default=448)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--no-stop-on-pass", dest="stop_on_pass", action="store_false")
    parser.set_defaults(stop_on_pass=True)
    parser.add_argument("--disable-incentive", action="store_true")
    parser.add_argument("--lambda-round", type=float, default=None)
    parser.add_argument("--lambda-token", type=float, default=None)
    parser.add_argument("--cost-token-mode", choices=("outputs_only", "context_and_outputs"), default=None)
    parser.add_argument("--difficulty-estimate-min", type=int, default=80)
    parser.add_argument("--difficulty-estimate-max", type=int, default=800)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--weight-min", type=float, default=None)
    parser.add_argument("--weight-max", type=float, default=None)
    parser.add_argument("--weight-power", type=float, default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(args.config)
    cost_cfg = cfg.table("cost")
    inc_cfg = cfg.table("incentive")
    pref_cfg = cfg.raw.get("preference", {})
    args.role_boundary_cfg = cfg.raw.get("role_boundary", {})
    exp_cfg = cfg.table("experiment")
    if args.model_path is None:
        args.model_path = str(cfg.table("model").get("local_model_path") or cfg.table("model")["backbone"])
    args.lambda_round = float(args.lambda_round if args.lambda_round is not None else cost_cfg["lambda_round"])
    args.lambda_token = float(args.lambda_token if args.lambda_token is not None else cost_cfg["lambda_token"])
    args.cost_token_mode = str(args.cost_token_mode or cost_cfg.get("leader_token_mode", "context_and_outputs"))
    args.max_rounds = int(args.max_rounds if args.max_rounds is not None else exp_cfg.get("max_rounds", 6))
    margin = float(args.margin if args.margin is not None else pref_cfg.get("leader_margin", 0.03))
    weight_min = float(args.weight_min if args.weight_min is not None else pref_cfg.get("weight_min", 0.3))
    weight_max = float(args.weight_max if args.weight_max is not None else pref_cfg.get("weight_max", 2.0))
    weight_power = float(args.weight_power if args.weight_power is not None else pref_cfg.get("weight_power", 0.65))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / f"{args.output_prefix}_trajectories.jsonl"
    preferences_path = output_dir / f"{args.output_prefix}_preferences.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    report_path = output_dir / f"{args.output_prefix}_weight_report.json"

    if args.shuffle_tasks:
        tasks = iter_tasks(cfg, args.split, None)
        random.Random(args.seed).shuffle(tasks)
        if args.limit is not None:
            tasks = tasks[: args.limit]
    else:
        tasks = iter_tasks(cfg, args.split, args.limit)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if args.num_shards > 1:
        selected_before_shard = len(tasks)
        tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_index]
        print(
            f"task shard {args.shard_index}/{args.num_shards}: {len(tasks)} of {selected_before_shard} selected tasks",
            flush=True,
        )
    planner_model, planner_tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.planner_adapter_path, args.load_in_8bit, args.load_in_4bit)
    if args.coder_adapter_path == args.planner_adapter_path:
        coder_model, coder_tokenizer = planner_model, planner_tokenizer
    else:
        coder_model, coder_tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.coder_adapter_path, args.load_in_8bit, args.load_in_4bit)
    executor = HumanEvalExecutor(cfg.phase1_dir)
    incentive_policy = str(inc_cfg.get("policy", "exp_token_difficulty"))
    incentive_enabled = bool(inc_cfg.get("enabled", True)) and not args.disable_incentive
    shuffled_budget_map = build_shuffled_budget_map(tasks, planner_tokenizer, inc_cfg, args) if incentive_enabled and incentive_policy == "shuffled_exp_token_difficulty" else None

    all_trajectories: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    trajectory_counter = 0

    with trajectories_path.open("w", encoding="utf-8") as traj_out, preferences_path.open("w", encoding="utf-8") as pref_out:
        for task_index, task in enumerate(tasks, start=1):
            task_trajectories = []
            print(f"[{task_index}/{len(tasks)}] leader-pref {task.task_id} {task.source_task_id} {task.entry_point}", flush=True)

            for sample_index, planner_temperature in enumerate(args.planner_temperatures):
                trajectory = run_trajectory(
                    planner_model,
                    planner_tokenizer,
                    coder_model,
                    coder_tokenizer,
                    executor,
                    task,
                    task_index,
                    sample_index,
                    planner_temperature,
                    args,
                    inc_cfg,
                    shuffled_budget_map,
                )
                trajectory["trajectory_id"] = f"{args.id_prefix}_{trajectory_counter:08d}"
                trajectory_counter += 1
                traj_out.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                traj_out.flush()
                all_trajectories.append(trajectory)
                task_trajectories.append(trajectory)
                diff_est = trajectory.get("difficulty", {}).get("difficulty_token_estimate", 0)
                print(
                    f"    temp={planner_temperature:.2f} passed={trajectory['passed']} "
                    f"r={trajectory['pass_rate']:.3f} rounds={trajectory['rounds']} "
                    f"diff_tok={diff_est:.0f} inc={trajectory['incentive_total']:.4f} "
                    f"u={trajectory['leader_utility']:.4f}",
                    flush=True,
                )

            pairs = make_preference_pairs(task_trajectories, margin, weight_min, weight_max, weight_power)
            for pair in pairs:
                pref_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pref_out.flush()
            all_pairs.extend(pairs)
            print(f"    pairs={len(pairs)}", flush=True)

    tasks_with_pairs = len({pair["task_id"] for pair in all_pairs})
    stats = weight_stats(all_pairs, weight_min, weight_max)
    summary = {
        "model_path": args.model_path,
        "planner_adapter_path": args.planner_adapter_path,
        "coder_adapter_path": args.coder_adapter_path,
        "split": args.split,
        "num_tasks": len(tasks),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_trajectories": len(all_trajectories),
        "num_preferences": len(all_pairs),
        "tasks_with_preferences": tasks_with_pairs,
        "tasks_without_preferences": len(tasks) - tasks_with_pairs,
        "planner_temperatures": args.planner_temperatures,
        "coder_temperature": args.coder_temperature,
        "max_rounds": args.max_rounds,
        "stop_on_pass": args.stop_on_pass,
        "lambda_round": args.lambda_round,
        "lambda_token": args.lambda_token,
        "cost_token_mode": args.cost_token_mode,
        "incentive_policy": "disabled" if not incentive_enabled else incentive_policy,
        "shuffled_incentive": bool(shuffled_budget_map),
        "shuffle_seed": inc_cfg.get("shuffle_seed"),
        "margin": margin,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "weight_power": weight_power,
        "trajectory_pass_rate_mean": sum(t["pass_rate"] for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0.0,
        "trajectory_passed_count": sum(1 for t in all_trajectories if t["passed"]),
        "leader_utility_mean": sum(t["leader_utility"] for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0.0,
        "avg_rounds": sum(t["rounds"] for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0.0,
        "avg_incentive_total": sum(t["incentive_total"] for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0.0,
        "avg_role_boundary_penalty_total": sum(t.get("role_boundary_penalty_total", 0.0) for t in all_trajectories) / len(all_trajectories) if all_trajectories else 0.0,
        "role_boundary_violation_trajectory_count": sum(1 for t in all_trajectories if t.get("role_boundary_penalty_total", 0.0) > 0),
        "role_boundary_config": args.role_boundary_cfg,
        "weight_stats": stats,
        "paths": {
            "trajectories": str(trajectories_path),
            "preferences": str(preferences_path),
            "summary": str(summary_path),
            "weight_report": str(report_path),
        },
    }
    report = {
        "summary": summary,
        "weights_reasonable_target": {
            "mean_around": 1.0,
            "preferred_range": [weight_min, weight_max],
            "desired_p90_lte": min(weight_max, 1.8),
            "desired_clipped_high_fraction_lte": 0.20,
        },
        "top_weight_pairs": sorted(
            [
                {
                    "task_id": p["task_id"],
                    "weight": p["weight"],
                    "raw_weight": p["raw_weight"],
                    "utility_delta": p["utility_delta"],
                    "chosen_utility": p["chosen_utility"],
                    "rejected_utility": p["rejected_utility"],
                }
                for p in all_pairs
            ],
            key=lambda x: x["weight"],
            reverse=True,
        )[:20],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

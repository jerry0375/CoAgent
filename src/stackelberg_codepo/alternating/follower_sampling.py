#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from stackelberg_codepo.config import ExperimentConfig as Config
from stackelberg_codepo.humaneval import HumanEvalExecutor, Task, load_tasks
from stackelberg_codepo.modeling.humaneval_model import extract_code, load_model_and_tokenizer


SYSTEM_PROMPT = (
    "You are the follower/coder in a planner-coder code generation team. "
    "Return only valid Python code. Do not include markdown fences, prose, tests, or examples."
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def load_task_map(cfg: Config) -> dict[str, Task]:
    tasks: dict[str, Task] = {}
    for path in (cfg.train_file, cfg.valid_file, cfg.task_file):
        for task in load_tasks(path):
            tasks[task.task_id] = task
    return tasks


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


def incentive_rule_text(total_budget: float, max_step_incentive: float, previous_best_pass_rate: float) -> str:
    return (
        "Incentive rule: execution improvements are rewarded by incremental pass-rate gain. "
        f"Remaining trajectory incentive budget is about {total_budget:.4f}; "
        f"per-step payment is capped at {max_step_incentive:.4f}; "
        f"current best pass rate is {previous_best_pass_rate:.3f}."
    )


def build_state_text(task: Task, plan: str, previous_code: str, previous_feedback: dict[str, Any] | None, incentive_text: str) -> str:
    parts = [
        f"Task ID: {task.task_id}",
        f"Entry point: {task.entry_point}",
        "Task prompt:",
        task.prompt,
    ]
    if previous_code:
        parts.extend(["Previous code:", previous_code])
    if previous_feedback:
        parts.extend(["Execution feedback:", compact_feedback(previous_feedback)])
    parts.extend(["Planner guidance:", plan, incentive_text])
    return "\n".join(parts)


def coder_messages(task: Task, state_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT + f" Define the function {task.entry_point} exactly."},
        {
            "role": "user",
            "content": (
                f"Task state:\n{state_text}\n\n"
                f"Write the complete Python function `{task.entry_point}`. Return Python code only."
            ),
        },
    ]


def sample_code(model, tokenizer, task: Task, state_text: str, temperature: float, top_p: float, seed: int, max_new_tokens: int):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    messages = coder_messages(task, state_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    code, code_type = extract_code(generated, task)
    return prompt, generated, code, code_type, elapsed


def trajectory_states(trajectories: list[dict[str, Any]], task_map: dict[str, Task], max_states: int, seed: int, prefer_repair: bool) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for traj in trajectories:
        task = task_map.get(traj.get("task_id"))
        if task is None:
            continue
        turns = traj.get("turns", [])
        best_before = 0.0
        previous_code = ""
        previous_feedback = None
        for idx, turn in enumerate(turns):
            round_index = int(turn.get("round", idx + 1))
            total_budget = float(traj.get("incentive_budget", 0.0))
            max_step = float(max(float(t.get("incentive", 0.0)) for t in turns) if turns else 0.0)
            if max_step <= 0:
                max_step = 0.08
            incentive_text = incentive_rule_text(total_budget, max_step, best_before)
            state_text = build_state_text(task, turn.get("plan", ""), previous_code, previous_feedback, incentive_text)
            states.append(
                {
                    "state_id": f"{traj.get('trajectory_id')}_round{round_index}",
                    "trajectory_id": traj.get("trajectory_id"),
                    "task_id": task.task_id,
                    "source_task_id": task.source_task_id,
                    "split": task.split,
                    "entry_point": task.entry_point,
                    "round": round_index,
                    "task": task,
                    "state_text": state_text,
                    "planner_instruction": turn.get("plan", ""),
                    "previous_best_pass_rate": best_before,
                    "incentive_budget": total_budget,
                    "max_step_incentive": max_step,
                    "incentive_rule": "exp_token_difficulty",
                    "previous_code": previous_code,
                    "previous_feedback": previous_feedback,
                }
            )
            best_before = max(best_before, clamp01(float(turn.get("pass_rate", 0.0))))
            previous_code = turn.get("code", "")
            previous_feedback = turn.get("feedback", {})

    rng = random.Random(seed)
    rng.shuffle(states)
    if prefer_repair:
        states.sort(key=lambda item: 0 if int(item["round"]) > 1 else 1)
    return states[:max_states]


def utility_from_candidate(candidate: dict[str, Any], beta_pass: float, beta_delta_pass: float, lambda_follower_token: float) -> float:
    r_pass = clamp01(candidate["pass_rate"])
    delta = max(0.0, r_pass - float(candidate["previous_best_pass_rate"]))
    incentive = min(float(candidate["max_step_incentive"]), float(candidate["incentive_budget"]) * delta)
    candidate["incentive"] = incentive
    candidate["delta_pass_rate"] = delta
    return beta_pass * r_pass + beta_delta_pass * delta - lambda_follower_token * candidate["code_tokens"] + incentive


def make_preference_pairs(
    candidates: list[dict[str, Any]],
    margin: float,
    weight_min: float,
    weight_max: float,
    weight_power: float,
) -> list[dict[str, Any]]:
    raw_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for a, b in itertools.combinations(candidates, 2):
        diff = float(a["utility"]) - float(b["utility"])
        if abs(diff) <= margin:
            continue
        chosen, rejected = (a, b) if diff > 0 else (b, a)
        raw_pairs.append((chosen, rejected, abs(diff)))
    if not raw_pairs:
        return []
    avg_delta = sum(delta for _, _, delta in raw_pairs) / len(raw_pairs)
    pairs: list[dict[str, Any]] = []
    for chosen, rejected, delta in raw_pairs:
        raw_weight = delta / (avg_delta + 1e-8)
        weight = max(weight_min, min(weight_max, raw_weight ** weight_power))
        pairs.append(
            {
                "task_id": chosen["task_id"],
                "source_task_id": chosen["source_task_id"],
                "state_id": chosen["state_id"],
                "trajectory_id": chosen["trajectory_id"],
                "round": chosen["round"],
                "planner_instruction": chosen["planner_instruction"],
                "incentive_rule": chosen["incentive_rule"],
                "chosen_candidate_id": chosen["candidate_id"],
                "rejected_candidate_id": rejected["candidate_id"],
                "chosen_code": chosen["code"],
                "rejected_code": rejected["code"],
                "chosen_utility": chosen["utility"],
                "rejected_utility": rejected["utility"],
                "utility_delta": delta,
                "raw_weight": raw_weight,
                "weight": weight,
                "chosen_metrics": chosen["metrics"],
                "rejected_metrics": rejected["metrics"],
                "state_tokens": chosen["state_tokens"],
                "chosen_code_tokens": chosen["code_tokens"],
                "rejected_code_tokens": rejected["code_tokens"],
            }
        )
    return pairs



def apply_global_weights(pairs: list[dict[str, Any]], weight_min: float, weight_max: float, weight_power: float) -> None:
    if not pairs:
        return
    avg_delta = sum(float(pair["utility_delta"]) for pair in pairs) / len(pairs)
    for pair in pairs:
        raw_weight = float(pair["utility_delta"]) / (avg_delta + 1e-8)
        pair["local_raw_weight"] = pair.get("raw_weight")
        pair["local_weight"] = pair.get("weight")
        pair["raw_weight"] = raw_weight
        pair["weight"] = max(weight_min, min(weight_max, raw_weight ** weight_power))

def weight_stats(pairs: list[dict[str, Any]]) -> dict[str, float | int]:
    weights = [float(p["weight"]) for p in pairs]
    if not weights:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "stdev": 0.0}
    return {
        "count": len(weights),
        "min": min(weights),
        "max": max(weights),
        "mean": statistics.mean(weights),
        "stdev": statistics.pstdev(weights) if len(weights) > 1 else 0.0,
    }


def write_wpo_jsonl(pairs: list[dict[str, Any]], candidates: dict[str, dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            chosen = candidates.get(pair["chosen_candidate_id"])
            if chosen is None:
                continue
            payload = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {
                        "from": "human",
                        "value": (
                            f"Task state:\n{chosen['state_text']}\n\n"
                            f"Write the complete Python function `{chosen['entry_point']}`. Return Python code only."
                        ),
                    },
                ],
                "chosen": {"from": "gpt", "value": pair["chosen_code"].rstrip() + "\n"},
                "rejected": {"from": "gpt", "value": pair["rejected_code"].rstrip() + "\n"},
                "weight": pair["weight"],
                "metadata": {
                    "task_id": pair["task_id"],
                    "state_id": pair["state_id"],
                    "round": pair["round"],
                    "utility_delta": pair["utility_delta"],
                    "chosen_utility": pair["chosen_utility"],
                    "rejected_utility": pair["rejected_utility"],
                    "raw_weight": pair["raw_weight"],
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build follower preferences from sampled states in multi-round leader trajectories.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_algorithm_smoke.json"))
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "preference_demo"))
    parser.add_argument("--output-prefix", default="follower_pref_from_traj")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-states", type=int, default=6)
    parser.add_argument("--prefer-repair-states", action="store_true")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.2, 0.6, 0.8, 1.0])
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--weight-min", type=float, default=None)
    parser.add_argument("--weight-max", type=float, default=None)
    parser.add_argument("--weight-power", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(args.config)
    cost_cfg = cfg.table("cost")
    pref_cfg = cfg.raw.get("preference", {})
    if args.model_path is None:
        args.model_path = str(cfg.table("model").get("local_model_path") or cfg.table("model")["backbone"])
    beta_pass = float(cost_cfg["beta_pass"])
    beta_delta_pass = float(cost_cfg["beta_delta_pass"])
    lambda_follower_token = float(cost_cfg["lambda_follower_token"])
    weight_min = float(args.weight_min if args.weight_min is not None else pref_cfg.get("weight_min", 0.3))
    weight_max = float(args.weight_max if args.weight_max is not None else pref_cfg.get("weight_max", 2.0))
    weight_power = float(args.weight_power if args.weight_power is not None else pref_cfg.get("weight_power", 0.65))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / f"{args.output_prefix}_candidates.jsonl"
    preferences_path = output_dir / f"{args.output_prefix}_preferences.jsonl"
    wpo_path = output_dir / f"{args.output_prefix}_wpo.jsonl"
    summary_path = output_dir / f"{args.output_prefix}_summary.json"

    trajectories = load_jsonl(Path(args.trajectories))
    task_map = load_task_map(cfg)
    states = trajectory_states(trajectories, task_map, args.limit_states, args.seed, args.prefer_repair_states)
    configured_max_step_incentive = float(cfg.table("incentive").get("max_step_incentive", 0.08))
    for state in states:
        state["max_step_incentive"] = configured_max_step_incentive
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    executor = HumanEvalExecutor(cfg.phase1_dir)

    all_candidates: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    candidate_counter = 0
    with candidates_path.open("w", encoding="utf-8") as cand_out, preferences_path.open("w", encoding="utf-8") as pref_out:
        for state_index, state in enumerate(states, start=1):
            task: Task = state["task"]
            state_tokens = token_count(tokenizer, state["state_text"])
            state_candidates: list[dict[str, Any]] = []
            print(f"[{state_index}/{len(states)}] follower-state {state['state_id']} {task.task_id} round={state['round']}", flush=True)
            for sample_index, temperature in enumerate(args.temperatures):
                seed = args.seed + state_index * 100 + sample_index
                prompt, generated, code, code_type, generation_time = sample_code(
                    model,
                    tokenizer,
                    task,
                    state["state_text"],
                    temperature,
                    args.top_p,
                    seed,
                    args.max_new_tokens,
                )
                feedback = executor.execute(task, code, code_type=code_type)
                code_tokens = token_count(tokenizer, code)
                candidate = {
                    "candidate_id": f"cand_{candidate_counter:08d}",
                    "task_id": task.task_id,
                    "source_task_id": task.source_task_id,
                    "split": task.split,
                    "entry_point": task.entry_point,
                    "state_id": state["state_id"],
                    "trajectory_id": state["trajectory_id"],
                    "round": state["round"],
                    "state_text": state["state_text"],
                    "state_tokens": state_tokens,
                    "planner_instruction": state["planner_instruction"],
                    "incentive_rule": state["incentive_rule"],
                    "incentive_budget": state["incentive_budget"],
                    "max_step_incentive": state["max_step_incentive"],
                    "previous_best_pass_rate": state["previous_best_pass_rate"],
                    "sample_index": sample_index,
                    "temperature": temperature,
                    "top_p": args.top_p,
                    "seed": seed,
                    "prompt_tokens_actual": token_count(tokenizer, prompt),
                    "generated_text": generated,
                    "code": code,
                    "code_type": code_type,
                    "code_tokens": code_tokens,
                    "generation_time": generation_time,
                    "passed": bool(feedback.passed),
                    "pass_rate_raw": float(feedback.raw.get("pass_rate", 0.0)),
                    "pass_rate": clamp01(float(feedback.raw.get("pass_rate", 0.0))),
                    "feedback": feedback.raw,
                }
                candidate["utility"] = utility_from_candidate(candidate, beta_pass, beta_delta_pass, lambda_follower_token)
                candidate["metrics"] = {
                    "passed": candidate["passed"],
                    "pass_rate": candidate["pass_rate"],
                    "pass_rate_raw": candidate["pass_rate_raw"],
                    "delta_pass_rate": candidate["delta_pass_rate"],
                    "incentive": candidate["incentive"],
                    "code_tokens": candidate["code_tokens"],
                    "utility": candidate["utility"],
                    "error_type": feedback.raw.get("error_type"),
                    "error_message": feedback.raw.get("error_message"),
                }
                cand_out.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                cand_out.flush()
                all_candidates.append(candidate)
                state_candidates.append(candidate)
                candidate_counter += 1
                print(
                    f"    temp={temperature:.2f} passed={candidate['passed']} r={candidate['pass_rate']:.3f} "
                    f"delta={candidate['delta_pass_rate']:.3f} inc={candidate['incentive']:.4f} "
                    f"tok={code_tokens} u={candidate['utility']:.4f}",
                    flush=True,
                )

            pairs = make_preference_pairs(state_candidates, args.margin, weight_min, weight_max, weight_power)
            for pair in pairs:
                pref_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            pref_out.flush()
            all_pairs.extend(pairs)
            print(f"    pairs={len(pairs)}", flush=True)

    apply_global_weights(all_pairs, weight_min, weight_max, weight_power)
    with preferences_path.open("w", encoding="utf-8") as pref_out:
        for pair in all_pairs:
            pref_out.write(json.dumps(pair, ensure_ascii=False) + "\n")
    candidate_by_id = {row["candidate_id"]: row for row in all_candidates}
    wpo_examples = write_wpo_jsonl(all_pairs, candidate_by_id, wpo_path)
    tasks_with_pairs = len({pair["task_id"] for pair in all_pairs})
    states_with_pairs = len({pair["state_id"] for pair in all_pairs})
    summary = {
        "model_path": args.model_path,
        "num_states": len(states),
        "num_candidates": len(all_candidates),
        "num_preferences": len(all_pairs),
        "num_wpo_examples": wpo_examples,
        "tasks_with_preferences": tasks_with_pairs,
        "states_with_preferences": states_with_pairs,
        "temperatures": args.temperatures,
        "beta_pass": beta_pass,
        "beta_delta_pass": beta_delta_pass,
        "lambda_follower_token": lambda_follower_token,
        "margin": args.margin,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "weight_power": weight_power,
        "candidate_pass_rate_mean": sum(c["pass_rate"] for c in all_candidates) / len(all_candidates) if all_candidates else 0.0,
        "candidate_passed_count": sum(1 for c in all_candidates if c["passed"]),
        "avg_incentive": sum(c.get("incentive", 0.0) for c in all_candidates) / len(all_candidates) if all_candidates else 0.0,
        "weight_stats": weight_stats(all_pairs),
        "paths": {
            "candidates": str(candidates_path),
            "preferences": str(preferences_path),
            "wpo": str(wpo_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

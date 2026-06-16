#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]

from stackelberg_codepo.config import ExperimentConfig as Config
from stackelberg_codepo.humaneval import HumanEvalExecutor, Task, load_tasks
from stackelberg_codepo.modeling.humaneval_model import extract_code, iter_tasks, load_model_and_tokenizer


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def compact_feedback(feedback_raw: dict, max_asserts: int = 8) -> str:
    fields = {
        "passed": feedback_raw.get("passed"),
        "pass_rate": feedback_raw.get("pass_rate"),
        "passed_tests": feedback_raw.get("passed_tests"),
        "total_tests": feedback_raw.get("total_tests"),
        "error_type": feedback_raw.get("error_type"),
        "error_message": feedback_raw.get("error_message"),
        "timeout": feedback_raw.get("timeout"),
    }
    failed = [item for item in feedback_raw.get("assert_results", []) if not item.get("passed")]
    if failed:
        fields["failed_asserts"] = failed[:max_asserts]
    stderr = (feedback_raw.get("stderr") or "").strip()
    if stderr:
        fields["stderr_tail"] = stderr.splitlines()[-8:]
    return json.dumps(fields, ensure_ascii=False, indent=2)


def chat_generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> tuple[str, float, int, int]:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = token_count(tokenizer, prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    elapsed = time.perf_counter() - start
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    completion_tokens = int(new_tokens.shape[-1])
    return generated, elapsed, prompt_tokens, completion_tokens


def incentive_rule_text(total_budget: float, max_step_incentive: float, best_pass_rate: float) -> str:
    return (
        "Incentive rule: execution improvements are rewarded by incremental pass-rate gain. "
        f"Remaining trajectory incentive budget is about {total_budget:.4f}; "
        f"per-step payment is capped at {max_step_incentive:.4f}; "
        f"current best pass rate is {best_pass_rate:.3f}."
    )


def planner_messages(
    task: Task,
    round_index: int,
    previous_code: str | None,
    previous_feedback: dict | None,
    prompt_profile: str,
) -> list[dict[str, str]]:
    if prompt_profile == "training_aligned":
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
                "The previous code did not fully solve the task. Diagnose the concrete defect from the failed assertions, "
                "then give the smallest correction the coder should make. Do not write code.\n"
                "Use this exact structure:\n"
                "Likely defect: <one sentence>\n"
                "Evidence: <cite the failed assertion or error>\n"
                "Minimal correction: <one sentence>\n"
                "Preserve: <behavior from the previous code that should not change>\n\n"
                f"Task:\n{task.prompt}\n\n"
                f"Previous code:\n{previous_code or ''}\n\n"
                f"Execution feedback:\n{compact_feedback(previous_feedback or {})}"
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    system = (
        "You are the planner in a planner-coder code generation team. "
        "Your job is to provide a concise implementation plan, key edge cases, and repair guidance. "
        "Do not write code. Keep the plan actionable and short."
    )
    if round_index == 1:
        user = (
            "Create an implementation plan for this HumanEval task. "
            "Mention edge cases the coder must handle.\n\n"
            f"{task.prompt}"
        )
    else:
        user = (
            "The previous code failed. Diagnose the concrete defect from the failed assertions, "
            "then give the smallest correction the coder should make. Do not write code.\n"
            "Use this exact structure:\n"
            "Likely defect: <one sentence>\n"
            "Evidence: <cite the failed assertion or error>\n"
            "Minimal correction: <one sentence>\n"
            "Preserve: <behavior from the previous code that should not change>\n\n"
            f"Task:\n{task.prompt}\n\n"
            f"Previous code:\n{previous_code or ''}\n\n"
            f"Execution feedback:\n{compact_feedback(previous_feedback or {})}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def coder_messages(
    task: Task,
    plan: str,
    round_index: int,
    previous_code: str | None,
    previous_feedback: dict | None,
    prompt_profile: str,
    incentive_text: str,
) -> list[dict[str, str]]:
    if prompt_profile == "training_aligned":
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
                "Now write a corrected complete Python function. Make the smallest necessary logical change and preserve behavior that already passed."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    system = (
        "You are the coder in a planner-coder code generation team. "
        "Return only valid Python code. Do not include markdown fences, prose, tests, or examples. "
        f"Define the function {task.entry_point} exactly."
    )
    if round_index == 1:
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Planner guidance:\n{plan}\n\n"
            "Now write the complete Python function."
        )
    else:
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Previous code:\n{previous_code or ''}\n\n"
            f"Execution feedback:\n{compact_feedback(previous_feedback or {})}\n\n"
            f"Planner repair guidance:\n{plan}\n\n"
            f"{incentive_text}\n\n"
            "Now write a corrected complete Python function. Make the smallest necessary logical change and preserve behavior that already passed."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def pass_rate_from_turn(turn: dict) -> float:
    return clamp01(float(turn.get("feedback", {}).get("pass_rate", 0.0)))


def turn_sort_key(turn: dict) -> tuple[float, int, int]:
    return (
        pass_rate_from_turn(turn),
        1 if bool(turn.get("feedback", {}).get("passed")) else 0,
        -int(turn.get("code_tokens", 0)),
    )


def run_task(planner_model, planner_tokenizer, coder_model, coder_tokenizer, executor: HumanEvalExecutor, task: Task, args: argparse.Namespace) -> dict:
    turns = []
    previous_code = None
    previous_feedback = None
    first_round_passed = False
    best_pass_rate = 0.0
    best_turn = None

    for round_index in range(1, args.max_rounds + 1):
        plan, planner_time, planner_prompt_tokens, planner_completion_tokens = chat_generate(
            planner_model,
            planner_tokenizer,
            planner_messages(task, round_index, previous_code, previous_feedback, args.prompt_profile),
            args.planner_max_new_tokens,
            args.temperature,
            args.top_p,
        )
        incentive_text = incentive_rule_text(args.demo_incentive_budget, args.demo_max_step_incentive, best_pass_rate)
        plan_tokens = token_count(planner_tokenizer, plan)
        sample_count = int(args.repair_num_samples) if round_index > 1 else 1
        coder_temperature = float(args.repair_temperature) if round_index > 1 and args.repair_temperature is not None else float(args.temperature)
        candidates = []
        for sample_index in range(sample_count):
            generated, coder_time, coder_prompt_tokens, coder_completion_tokens = chat_generate(
                coder_model,
                coder_tokenizer,
                coder_messages(task, plan, round_index, previous_code, previous_feedback, args.prompt_profile, incentive_text),
                args.coder_max_new_tokens,
                coder_temperature,
                args.top_p,
            )
            code, code_type = extract_code(generated, task)
            code_tokens = token_count(coder_tokenizer, code)
            feedback = executor.execute(task, code, code_type=code_type)
            candidates.append(
                {
                    "round": round_index,
                    "repair_sample_index": sample_index,
                    "plan": plan,
                    "planner_time": planner_time if sample_index == 0 else 0.0,
                    "planner_prompt_tokens": planner_prompt_tokens if sample_index == 0 else 0,
                    "planner_completion_tokens": planner_completion_tokens if sample_index == 0 else 0,
                    "plan_tokens": plan_tokens if sample_index == 0 else 0,
                    "generated_text": generated,
                    "code": code,
                    "code_type": code_type,
                    "coder_time": coder_time,
                    "coder_prompt_tokens": coder_prompt_tokens,
                    "coder_completion_tokens": coder_completion_tokens,
                    "code_tokens": code_tokens,
                    "feedback": feedback.raw,
                }
            )

        turn = dict(max(candidates, key=turn_sort_key))
        if len(candidates) > 1:
            turn["repair_candidate_count"] = len(candidates)
            turn["coder_time_total"] = sum(float(candidate.get("coder_time", 0.0)) for candidate in candidates)
            turn["coder_prompt_tokens_total"] = sum(int(candidate.get("coder_prompt_tokens", 0)) for candidate in candidates)
            turn["coder_completion_tokens_total"] = sum(int(candidate.get("coder_completion_tokens", 0)) for candidate in candidates)
            turn["repair_candidates"] = [dict(candidate) for candidate in candidates]
        if round_index == 1:
            first_round_passed = bool(turn["feedback"].get("passed"))
        turns.append(turn)
        if best_turn is None or turn_sort_key(turn) > turn_sort_key(best_turn):
            best_turn = turn
        context_turn = best_turn if args.best_so_far else turn
        previous_code = context_turn["code"]
        previous_feedback = context_turn["feedback"]
        best_pass_rate = max(best_pass_rate, pass_rate_from_turn(turn))
        if bool(turn["feedback"].get("passed")):
            break

    final_turn = best_turn if args.best_so_far and best_turn is not None else (turns[-1] if turns else {})
    final_feedback = final_turn.get("feedback", {}) if final_turn else {}
    return {
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "split": task.split,
        "entry_point": task.entry_point,
        "first_round_passed": first_round_passed,
        "final_passed": bool(final_feedback.get("passed")),
        "rounds_used": len(turns),
        "final_pass_rate": float(final_feedback.get("pass_rate", 0.0)),
        "final_selected_round": final_turn.get("round") if final_turn else None,
        "best_so_far_enabled": bool(args.best_so_far),
        "repair_num_samples": int(args.repair_num_samples),
        "turns": turns,
    }


def read_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    items = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items[item["task_id"]] = item
    return items


def summarize(records: list[dict], args: argparse.Namespace) -> dict:
    total = len(records)
    first_passed = sum(1 for item in records if item.get("first_round_passed"))
    final_passed = sum(1 for item in records if item.get("final_passed"))
    rounds = [int(item.get("rounds_used", 0)) for item in records]
    by_split = Counter(item.get("split", "") for item in records)
    planner_tokens = []
    coder_tokens = []
    total_tokens = []
    costs = []
    utilities = []
    lambda_round = float(getattr(args, "lambda_round", 0.02))
    lambda_token = float(getattr(args, "lambda_token", 0.00001))
    for item in records:
        p_tok = sum(int(turn.get("planner_prompt_tokens", 0)) + int(turn.get("planner_completion_tokens", 0)) for turn in item.get("turns", []))
        c_tok = 0
        for turn in item.get("turns", []):
            c_tok += int(turn.get("coder_prompt_tokens_total", turn.get("coder_prompt_tokens", 0)))
            c_tok += int(turn.get("coder_completion_tokens_total", turn.get("coder_completion_tokens", 0)))
        tok = p_tok + c_tok
        rounds_used = int(item.get("rounds_used", 0))
        quality = clamp01(float(item.get("final_pass_rate", 0.0)))
        cost = lambda_round * rounds_used + lambda_token * tok
        planner_tokens.append(p_tok)
        coder_tokens.append(c_tok)
        total_tokens.append(tok)
        costs.append(cost)
        utilities.append(quality - cost)
    return {
        "model_path": args.model_path,
        "split": args.split,
        "num_tasks": total,
        "first_round_passed": first_passed,
        "first_round_pass_at_1": first_passed / total if total else 0.0,
        "final_passed": final_passed,
        "final_pass_rate_binary": final_passed / total if total else 0.0,
        "avg_assert_pass_rate_final": sum(float(item.get("final_pass_rate", 0.0)) for item in records) / total if total else 0.0,
        "avg_rounds_used": sum(rounds) / total if total else 0.0,
        "max_rounds": args.max_rounds,
        "by_split": dict(by_split),
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "coder_max_new_tokens": args.coder_max_new_tokens,
        "temperature": args.temperature,
        "prompt_profile": getattr(args, "prompt_profile", "legacy"),
        "demo_incentive_budget": getattr(args, "demo_incentive_budget", None),
        "demo_max_step_incentive": getattr(args, "demo_max_step_incentive", None),
        "lambda_round": lambda_round,
        "lambda_token": lambda_token,
        "avg_planner_tokens": sum(planner_tokens) / total if total else 0.0,
        "avg_coder_tokens": sum(coder_tokens) / total if total else 0.0,
        "avg_total_tokens": sum(total_tokens) / total if total else 0.0,
        "avg_total_cost": sum(costs) / total if total else 0.0,
        "avg_leader_utility": sum(utilities) / total if total else 0.0,
        "planner_adapter_path": getattr(args, "planner_adapter_path", None),
        "coder_adapter_path": getattr(args, "coder_adapter_path", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo planner-coder role collaboration on HumanEval.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_algorithm_smoke.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--planner-adapter-path", default=None, help="Optional PEFT/LoRA adapter path for planner generation.")
    parser.add_argument("--coder-adapter-path", default=None, help="Optional PEFT/LoRA adapter path for coder generation.")
    parser.add_argument("--split", choices=("test", "valid", "train", "all"), default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "role_demo"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--planner-max-new-tokens", type=int, default=192)
    parser.add_argument("--coder-max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-profile", choices=("legacy", "training_aligned"), default="legacy")
    parser.add_argument("--best-so-far", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--repair-num-samples", type=int, default=1)
    parser.add_argument("--repair-temperature", type=float, default=None)
    parser.add_argument("--demo-incentive-budget", type=float, default=None)
    parser.add_argument("--demo-max-step-incentive", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(args.config)
    if args.model_path is None:
        args.model_path = str(cfg.table("model").get("local_model_path") or cfg.table("model")["backbone"])
    cost_cfg = cfg.table("cost")
    args.lambda_round = float(cost_cfg.get("lambda_round", 0.02))
    args.lambda_token = float(cost_cfg.get("lambda_token", 0.00001))
    inc_cfg = cfg.table("incentive")
    if args.demo_incentive_budget is None:
        args.demo_incentive_budget = float(inc_cfg.get("total_budget_base", 0.035))
    if args.demo_max_step_incentive is None:
        args.demo_max_step_incentive = float(inc_cfg.get("max_step_incentive", 0.08))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name.replace("/", "_")
    output_name = args.output_name or f"{model_name}_{args.split}_role_demo"
    result_path = output_dir / f"{output_name}.jsonl"
    summary_path = output_dir / f"{output_name}_summary.json"

    tasks = iter_tasks(cfg, args.split, args.limit)
    existing = {} if args.no_resume else read_existing(result_path)
    planner_model, planner_tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.planner_adapter_path)
    if args.coder_adapter_path == args.planner_adapter_path:
        coder_model, coder_tokenizer = planner_model, planner_tokenizer
    else:
        coder_model, coder_tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.coder_adapter_path)
    executor = HumanEvalExecutor(cfg.phase1_dir)

    completed = dict(existing)
    mode = "a" if existing and not args.no_resume else "w"
    with result_path.open(mode, encoding="utf-8") as out:
        for index, task in enumerate(tasks, start=1):
            if task.task_id in completed:
                print(f"[{index}/{len(tasks)}] skip {task.task_id}", flush=True)
                continue
            print(f"[{index}/{len(tasks)}] role-demo {task.task_id} {task.source_task_id} {task.entry_point}", flush=True)
            item = run_task(planner_model, planner_tokenizer, coder_model, coder_tokenizer, executor, task, args)
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            completed[task.task_id] = item
            print(
                f"    first={item['first_round_passed']} final={item['final_passed']} "
                f"rounds={item['rounds_used']} pass_rate={item['final_pass_rate']:.3f}",
                flush=True,
            )

    ordered = [completed[task.task_id] for task in tasks if task.task_id in completed]
    summary = summarize(ordered, args)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"results: {result_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

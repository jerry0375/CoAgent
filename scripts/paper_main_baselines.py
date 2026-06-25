#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stackelberg_codepo.config import ExperimentConfig as Config
from stackelberg_codepo.humaneval import HumanEvalExecutor, Task
from stackelberg_codepo.modeling.humaneval_model import extract_code, iter_tasks, load_model_and_tokenizer


METHODS = [
    "direct_prompt",
    "cot",
    "self_consistency_3",
    "self_repair",
    "mad",
    "agentcoder",
]

DISPLAY_NAMES = {
    "direct_prompt": "Direct Prompt",
    "cot": "CoT",
    "self_consistency_3": "Self-Consistency@3",
    "self_repair": "Self-Repair",
    "mad": "MAD",
    "agentcoder": "AgentCoder",
}


def token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def feedback_dict(feedback) -> dict:
    payload = asdict(feedback)
    payload.pop("raw", None)
    return payload


def compact_feedback(feedback: dict, max_asserts: int = 8) -> str:
    fields = {
        "passed": feedback.get("passed"),
        "pass_rate": feedback.get("pass_rate"),
        "passed_tests": feedback.get("passed_tests"),
        "total_tests": feedback.get("total_tests"),
        "error_type": feedback.get("error_type"),
        "error_message": feedback.get("error_message"),
        "timeout": feedback.get("timeout"),
    }
    failed = [item for item in feedback.get("raw", {}).get("assert_results", []) if not item.get("passed")]
    if failed:
        fields["failed_asserts"] = failed[:max_asserts]
    return json.dumps(fields, ensure_ascii=False, indent=2)


def _merge_system_into_user(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    system_text = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system").strip()
    merged = [dict(m) for m in messages if m.get("role") != "system"]
    if not system_text:
        return merged
    if merged and merged[0].get("role") == "user":
        merged[0]["content"] = system_text + "\n\n" + merged[0].get("content", "")
    else:
        merged.insert(0, {"role": "user", "content": system_text})
    return merged


def chat_generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> dict:
    import torch

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        if "System role not supported" not in str(exc):
            raise
        prompt = tokenizer.apply_chat_template(_merge_system_into_user(messages), tokenize=False, add_generation_prompt=True)
    prompt_tokens = token_count(tokenizer, prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs = {
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
    return {
        "text": generated,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(new_tokens.shape[-1]),
        "elapsed": elapsed,
    }


def code_messages(task: Task, style: str, extra: str = "") -> list[dict[str, str]]:
    if style == "cot":
        system = (
            "You are an expert Python programmer. Think through the algorithm briefly, "
            "then provide the final complete Python function. The final answer must contain valid Python code."
        )
        user = (
            f"Solve this HumanEval problem. The required function name is {task.entry_point}.\n\n"
            f"{task.prompt}\n\n"
            "Use this format:\nReasoning: <brief reasoning>\nFinal code:\n```python\n<code>\n```"
        )
    else:
        system = (
            "You are an expert Python programmer. Return only valid Python code. "
            "Do not include markdown fences, prose, tests, or examples."
        )
        user = (
            "Solve this HumanEval problem. Provide a complete Python function named "
            f"{task.entry_point}. Return only the code.\n\n{task.prompt}"
        )
    if extra:
        user += "\n\n" + extra
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def repair_messages(task: Task, previous_code: str, previous_feedback: dict, style: str = "self_repair") -> list[dict[str, str]]:
    if style == "agentcoder":
        system = (
            "You are the debugger in an AgentCoder-style team. Return only corrected Python code. "
            "Do not include markdown fences, prose, tests, or examples."
        )
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Programmer code:\n{previous_code}\n\n"
            f"Tester/debug feedback:\n{compact_feedback(previous_feedback)}\n\n"
            "Repair the implementation. Keep correct behavior and make the smallest necessary change."
        )
    else:
        system = (
            "You are an expert Python debugger. Return only corrected Python code. "
            "Do not include markdown fences, prose, tests, or examples."
        )
        user = (
            f"Task:\n{task.prompt}\n\n"
            f"Previous code:\n{previous_code}\n\n"
            f"Execution feedback:\n{compact_feedback(previous_feedback)}\n\n"
            "Write a corrected complete Python function."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def mad_plan_messages(task: Task, persona: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                f"You are {persona} in a multi-agent debate for Python code generation. "
                "Do not write code. Give concise algorithmic advice and edge cases."
            ),
        },
        {"role": "user", "content": f"Analyze this HumanEval task for implementation:\n\n{task.prompt}"},
    ]


def mad_final_messages(task: Task, plans: list[str]) -> list[dict[str, str]]:
    joined = "\n\n".join(f"Agent {i + 1}:\n{plan}" for i, plan in enumerate(plans))
    return [
        {
            "role": "system",
            "content": (
                "You are the final coder after a multi-agent debate. Return only valid Python code. "
                "Do not include markdown fences, prose, tests, or examples."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task:\n{task.prompt}\n\n"
                f"Debate notes:\n{joined}\n\n"
                f"Now write the complete Python function named {task.entry_point}."
            ),
        },
    ]


def agentcoder_test_messages(task: Task, code: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the tester in an AgentCoder-style team. Do not write final code. "
                "Identify likely edge cases and possible defects in the programmer's solution."
            ),
        },
        {
            "role": "user",
            "content": f"Task:\n{task.prompt}\n\nProgrammer code:\n{code}\n\nGive concise tester feedback.",
        },
    ]


def evaluate_generation(executor: HumanEvalExecutor, task: Task, generated: str) -> dict:
    code, code_type = extract_code(generated, task)
    feedback = executor.execute(task, code, code_type)
    return {"code": code, "code_type": code_type, "feedback": feedback_dict(feedback)}


def best_candidate(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda item: (
            float(item["feedback"].get("pass_rate", 0.0)),
            1 if item["feedback"].get("passed") else 0,
            -int(item.get("tokens", 0)),
        ),
    )


def run_one_method(model, tokenizer, executor: HumanEvalExecutor, task: Task, method: str, args: argparse.Namespace) -> dict:
    generations: list[dict] = []
    candidates: list[dict] = []

    def generate(messages, temperature: float, max_new_tokens: int | None = None) -> dict:
        gen = chat_generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=max_new_tokens or args.max_new_tokens,
            temperature=temperature,
            top_p=args.top_p,
        )
        generations.append(gen)
        return gen

    if method == "direct_prompt":
        gen = generate(code_messages(task, "direct"), args.temperature)
        item = evaluate_generation(executor, task, gen["text"])
        candidates.append({**item, "tokens": gen["prompt_tokens"] + gen["completion_tokens"]})

    elif method == "cot":
        gen = generate(code_messages(task, "cot"), args.temperature)
        item = evaluate_generation(executor, task, gen["text"])
        candidates.append({**item, "tokens": gen["prompt_tokens"] + gen["completion_tokens"]})

    elif method == "self_consistency_3":
        for i in range(3):
            extra = f"Candidate {i + 1}: solve independently; prefer a simple robust implementation."
            gen = generate(code_messages(task, "direct", extra=extra), args.sample_temperature)
            item = evaluate_generation(executor, task, gen["text"])
            candidates.append({**item, "tokens": gen["prompt_tokens"] + gen["completion_tokens"], "sample_index": i})

    elif method == "self_repair":
        gen = generate(code_messages(task, "direct"), args.temperature)
        first = evaluate_generation(executor, task, gen["text"])
        first["tokens"] = gen["prompt_tokens"] + gen["completion_tokens"]
        candidates.append(first)
        if not first["feedback"].get("passed"):
            repair = generate(repair_messages(task, first["code"], first["feedback"]), args.temperature)
            second = evaluate_generation(executor, task, repair["text"])
            second["tokens"] = first["tokens"] + repair["prompt_tokens"] + repair["completion_tokens"]
            candidates.append(second)

    elif method == "mad":
        personas = ["an algorithm designer", "an edge-case reviewer", "a minimalist Python implementer"]
        plans = []
        for persona in personas:
            gen = generate(mad_plan_messages(task, persona), args.sample_temperature, max_new_tokens=args.plan_max_new_tokens)
            plans.append(gen["text"])
        final = generate(mad_final_messages(task, plans), args.temperature)
        item = evaluate_generation(executor, task, final["text"])
        item["tokens"] = sum(g["prompt_tokens"] + g["completion_tokens"] for g in generations)
        candidates.append(item)

    elif method == "agentcoder":
        gen = generate(code_messages(task, "direct", extra="You are the programmer role in an AgentCoder-style team."), args.temperature)
        first = evaluate_generation(executor, task, gen["text"])
        first["tokens"] = gen["prompt_tokens"] + gen["completion_tokens"]
        candidates.append(first)
        tester = generate(agentcoder_test_messages(task, first["code"]), args.sample_temperature, max_new_tokens=args.plan_max_new_tokens)
        if not first["feedback"].get("passed"):
            merged_feedback = dict(first["feedback"])
            merged_feedback["tester_feedback"] = tester["text"]
            repair = generate(repair_messages(task, first["code"], merged_feedback, style="agentcoder"), args.temperature)
            second = evaluate_generation(executor, task, repair["text"])
            second["tokens"] = first["tokens"] + tester["prompt_tokens"] + tester["completion_tokens"] + repair["prompt_tokens"] + repair["completion_tokens"]
            candidates.append(second)
        else:
            first["tokens"] += tester["prompt_tokens"] + tester["completion_tokens"]

    else:
        raise ValueError(f"Unsupported method: {method}")

    selected = best_candidate(candidates)
    total_tokens = sum(g["prompt_tokens"] + g["completion_tokens"] for g in generations)
    return {
        "task_id": task.task_id,
        "source_task_id": task.source_task_id,
        "split": task.split,
        "entry_point": task.entry_point,
        "method": method,
        "method_display": DISPLAY_NAMES[method],
        "generations": [
            {
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "elapsed": g["elapsed"],
            }
            for g in generations
        ],
        "num_generations": len(generations),
        "total_tokens": total_tokens,
        "final_code": selected["code"],
        "final_code_type": selected["code_type"],
        "feedback": selected["feedback"],
        "candidates": [
            {
                "sample_index": c.get("sample_index"),
                "tokens": c.get("tokens"),
                "code_type": c["code_type"],
                "feedback": c["feedback"],
            }
            for c in candidates
        ],
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict], args: argparse.Namespace) -> dict:
    total = len(records)
    passed = sum(1 for item in records if item["feedback"].get("passed"))
    avg_assert = sum(float(item["feedback"].get("pass_rate", 0.0)) for item in records) / total if total else 0.0
    avg_tokens = sum(int(item.get("total_tokens", 0)) for item in records) / total if total else 0.0
    by_split = Counter(item.get("split", "") for item in records)
    return {
        "method": records[0]["method"] if records else None,
        "method_display": records[0]["method_display"] if records else None,
        "model_path": args.model_path,
        "split": args.split,
        "num_tasks": total,
        "final_passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_assert_pass_rate": avg_assert,
        "avg_total_tokens": avg_tokens,
        "temperature": args.temperature,
        "sample_temperature": args.sample_temperature,
        "top_p": args.top_p,
        "by_split": dict(by_split),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper main-comparison baselines on HumanEval.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "strict_final_leader_soft_train64.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--split", choices=("test", "valid", "train", "all"), default="test")
    parser.add_argument("--limit", type=int, default=17)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "paper_main_baselines_a800"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--plan-max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    cfg = Config(args.config)
    if args.model_path is None:
        args.model_path = str(cfg.table("model").get("local_model_path") or cfg.table("model").get("model_path") or cfg.table("model")["backbone"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = HumanEvalExecutor(cfg.phase1_dir)
    tasks = iter_tasks(cfg, args.split, args.limit)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, adapter_path=None)

    all_summaries = []
    for method in args.methods:
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        result_path = method_dir / f"{method}.jsonl"
        summary_path = method_dir / f"{method}_summary.json"
        if result_path.exists() and not args.no_resume:
            records = []
            with result_path.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            done = {item["task_id"] for item in records}
        else:
            records = []
            done = set()

        for index, task in enumerate(tasks, start=1):
            if task.task_id in done:
                continue
            record = run_one_method(model, tokenizer, executor, task, method, args)
            records.append(record)
            write_jsonl(result_path, records)
            fb = record["feedback"]
            print(
                f"[{method} {index}/{len(tasks)}] {task.task_id} "
                f"passed={fb.get('passed')} pass_rate={float(fb.get('pass_rate', 0.0)):.3f} "
                f"tokens={record['total_tokens']}",
                flush=True,
            )

        summary = summarize(records, args)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_summaries.append(summary)

    table = []
    for item in all_summaries:
        table.append(
            {
                "Method": item["method_display"],
                "Pass Rate": round(100.0 * item["pass_rate"], 2),
                "Assert Pass Rate": round(100.0 * item["avg_assert_pass_rate"], 2),
                "Avg. Tokens": round(item["avg_total_tokens"]),
                "final_passed": item["final_passed"],
                "num_tasks": item["num_tasks"],
            }
        )
    (output_dir / "main_baselines_summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "main_baselines_summary.md").open("w", encoding="utf-8") as f:
        f.write("| Method | Pass Rate | Assert Pass Rate | Avg. Tokens | final_passed | num_tasks |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in table:
            f.write(
                f"| {row['Method']} | {row['Pass Rate']:.2f} | {row['Assert Pass Rate']:.2f} | "
                f"{row['Avg. Tokens']} | {row['final_passed']} | {row['num_tasks']} |\n"
            )
    print((output_dir / "main_baselines_summary.md").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

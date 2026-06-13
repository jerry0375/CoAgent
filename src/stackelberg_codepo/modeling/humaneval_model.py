#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[3]

from stackelberg_codepo.config import ExperimentConfig as Config
from stackelberg_codepo.humaneval import HumanEvalExecutor, Task, load_tasks


def iter_tasks(cfg: Config, split: str, limit: int | None) -> list[Task]:
    if split == "test":
        files = [cfg.task_file]
    elif split == "train":
        files = [cfg.train_file]
    elif split == "valid":
        files = [cfg.valid_file]
    elif split == "all":
        files = [cfg.train_file, cfg.valid_file, cfg.task_file]
    else:
        raise ValueError(f"Unsupported split: {split}")

    tasks: list[Task] = []
    for file in files:
        remaining = None if limit is None else max(0, limit - len(tasks))
        if remaining == 0:
            break
        tasks.extend(load_tasks(file, limit=remaining))
    return tasks


def read_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records[item["task_id"]] = item
    return records


def default_planner_instruction(task: Task) -> str:
    return (
        f"Implement `{task.entry_point}` exactly as specified. "
        "Handle edge cases from the docstring. Return concise Python code only."
    )


def build_follower_state_text(task: Task, planner_instruction: str) -> str:
    return "\n".join(
        [
            f"Task ID: {task.task_id}",
            f"Entry point: {task.entry_point}",
            "Task prompt:",
            task.prompt,
            "Planner instruction:",
            planner_instruction,
        ]
    )


def build_messages(task: Task, prompt_style: str = "single") -> list[dict[str, str]]:
    if prompt_style == "follower":
        planner_instruction = default_planner_instruction(task)
        state_text = build_follower_state_text(task, planner_instruction)
        return [
            {
                "role": "system",
                "content": (
                    "You are the coder in a planner-coder system. "
                    "Return only valid Python code. Do not include markdown fences, prose, tests, or examples."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task state:\n{state_text}\n\n"
                    "Incentive rule:\nnone_demo\n\n"
                    "Write the complete Python function requested by the task. Return Python code only."
                ),
            },
        ]

    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. Return only valid Python code. "
                "Do not use markdown fences, explanations, tests, or comments outside the solution."
            ),
        },
        {
            "role": "user",
            "content": (
                "Solve this HumanEval problem. Provide a complete Python function named "
                f"{task.entry_point}. Return only the code.\n\n"
                f"{task.prompt}"
            ),
        },
    ]


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def prompt_import_prelude(prompt: str) -> str:
    imports: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            break
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(line)
    return "\n".join(imports).strip()


def extract_code(generated: str, task: Task) -> tuple[str, str]:
    text = strip_code_fence(generated)
    # Drop common preambles if the model still emits prose.
    marker = f"def {task.entry_point}"
    idx = text.find(marker)
    if idx >= 0:
        text = text[idx:]
        code = text.strip() + "\n"
        prelude = prompt_import_prelude(task.prompt)
        if prelude and prelude not in code[:500]:
            code = prelude + "\n\n" + code
        return code, "full_code"

    # Fallback: if any function definition exists, treat it as full code.
    any_def = re.search(r"(^|\n)def\s+\w+\s*\(", text)
    if any_def:
        text = text[any_def.start():]
        code = text.strip() + "\n"
        prelude = prompt_import_prelude(task.prompt)
        if prelude and prelude not in code[:500]:
            code = prelude + "\n\n" + code
        return code, "full_code"

    # Last fallback: let the Phase 1 executor append it to the HumanEval prompt.
    return text.rstrip() + "\n", "completion"


def generate_one(model, tokenizer, task: Task, args) -> tuple[str, str, str, float]:
    import torch

    messages = build_messages(task, args.prompt_style)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    code, code_type = extract_code(generated, task)
    return generated, code, code_type, elapsed


def load_model_and_tokenizer(model_path: str, device: str, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def write_summary(path: Path, records: Iterable[dict], args: argparse.Namespace) -> dict:
    items = list(records)
    passed = sum(1 for item in items if item["feedback"].get("passed"))
    total = len(items)
    avg_pass_rate = sum(float(item["feedback"].get("pass_rate", 0.0)) for item in items) / total if total else 0.0
    by_split = Counter(item.get("split", "") for item in items)
    summary = {
        "model_path": args.model_path,
        "split": args.split,
        "num_tasks": total,
        "passed": passed,
        "pass_at_1": passed / total if total else 0.0,
        "avg_assert_pass_rate": avg_pass_rate,
        "by_split": dict(by_split),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "adapter_path": getattr(args, "adapter_path", None),
        "prompt_style": getattr(args, "prompt_style", "single"),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local Coder model on HumanEval with Phase 1 executor.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_algorithm_smoke.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None, help="Optional PEFT/LoRA adapter path to load on top of the base model.")
    parser.add_argument("--split", choices=("test", "valid", "train", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "humaneval_eval"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-style", choices=("single", "follower"), default="single")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(args.config)
    if args.model_path is None:
        args.model_path = str(cfg.table("model").get("local_model_path") or cfg.table("model")["backbone"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name.replace("/", "_")
    output_name = args.output_name or f"{model_name}_{args.split}"
    result_path = output_dir / f"{output_name}.jsonl"
    summary_path = output_dir / f"{output_name}_summary.json"

    tasks = iter_tasks(cfg, args.split, args.limit)
    existing = {} if args.no_resume else read_existing_results(result_path)
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, args.adapter_path)
    executor = HumanEvalExecutor(cfg.phase1_dir)

    mode = "a" if existing and not args.no_resume else "w"
    completed = dict(existing)
    with result_path.open(mode, encoding="utf-8") as out:
        for index, task in enumerate(tasks, start=1):
            if task.task_id in completed:
                print(f"[{index}/{len(tasks)}] skip {task.task_id} existing", flush=True)
                continue
            print(f"[{index}/{len(tasks)}] generate {task.task_id} {task.source_task_id} {task.entry_point}", flush=True)
            generated, code, code_type, generation_time = generate_one(model, tokenizer, task, args)
            feedback = executor.execute(task, code, code_type=code_type)
            item = {
                "task_id": task.task_id,
                "source_task_id": task.source_task_id,
                "split": task.split,
                "entry_point": task.entry_point,
                "generated_text": generated,
                "code": code,
                "code_type": code_type,
                "generation_time": generation_time,
                "feedback": feedback.raw,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            completed[task.task_id] = item
            print(
                f"    passed={feedback.passed} pass_rate={feedback.pass_rate:.3f} "
                f"tests={feedback.passed_tests}/{feedback.total_tests} gen_time={generation_time:.2f}s",
                flush=True,
            )

    ordered = [completed[task.task_id] for task in tasks if task.task_id in completed]
    summary = write_summary(summary_path, ordered, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"results: {result_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

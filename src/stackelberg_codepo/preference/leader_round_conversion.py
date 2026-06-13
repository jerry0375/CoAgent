#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are the leader/planner in a planner-coder code generation team. "
    "Give concise, actionable guidance. Do not write code."
)


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


def build_initial_prompt(state_text: str) -> str:
    return (
        f"Task state:\n{state_text}\n\n"
        "Create an implementation plan for the coder.\n"
        "Start with exactly one line: Estimated effort tokens: N\n"
        "Then mention the core algorithm and key edge cases. Do not write code."
    )


def compact_feedback(feedback: dict[str, Any]) -> str:
    fields = {
        "passed": feedback.get("passed"),
        "pass_rate": feedback.get("pass_rate"),
        "passed_tests": feedback.get("passed_tests"),
        "total_tests": feedback.get("total_tests"),
        "timeout": feedback.get("timeout"),
        "error_type": feedback.get("error_type"),
        "error_message": feedback.get("error_message"),
    }
    failed = [item for item in feedback.get("assert_results", []) if not item.get("passed")]
    if failed:
        fields["failed_asserts"] = failed[:6]
    return json.dumps(fields, ensure_ascii=False, indent=2)


def build_repair_prompt(traj: dict[str, Any], turn_index: int) -> str:
    prev = traj["turns"][turn_index - 1]
    state_text = traj.get("state_text", "")
    return (
        f"Task state:\n{state_text}\n\n"
        f"Previous code:\n{prev.get('code', '')}\n\n"
        f"Execution feedback:\n{compact_feedback(prev.get('feedback', {}))}\n\n"
        "Give a concise repair plan for the coder. Do not write code."
    )


def repair_state_signature(traj: dict[str, Any], turn_index: int) -> str:
    prev = traj["turns"][turn_index - 1]
    payload = {
        "task_id": traj.get("task_id"),
        "previous_code": prev.get("code", ""),
        "previous_feedback": compact_feedback(prev.get("feedback", {})),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def to_sharegpt(conversations: list[dict[str, str]], chosen: str, rejected: str) -> dict[str, Any]:
    return {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": chosen.strip() + "\n"},
        "rejected": {"from": "gpt", "value": rejected.strip() + "\n"},
    }


def make_initial_example(pair: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any]) -> dict[str, Any] | None:
    chosen_plan = chosen.get("turns", [{}])[0].get("plan", "")
    rejected_plan = rejected.get("turns", [{}])[0].get("plan", "")
    if not chosen_plan or not rejected_plan or chosen_plan == rejected_plan:
        return None
    conversations = [
        {"from": "system", "value": SYSTEM_PROMPT},
        {"from": "human", "value": build_initial_prompt(chosen.get("state_text", ""))},
    ]
    item = to_sharegpt(conversations, chosen_plan, rejected_plan)
    item["metadata"] = {
        "stage": "leader_initial",
        "task_id": pair.get("task_id"),
        "chosen_trajectory_id": pair.get("chosen_trajectory_id"),
        "rejected_trajectory_id": pair.get("rejected_trajectory_id"),
        "utility_delta": pair.get("utility_delta"),
        "chosen_utility": pair.get("chosen_utility"),
        "rejected_utility": pair.get("rejected_utility"),
        "weight": pair.get("weight", 1.0),
    }
    return item


def make_repair_examples_strict(pair: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    examples: list[dict[str, Any]] = []
    skipped = 0
    chosen_turns = chosen.get("turns", [])
    rejected_turns = rejected.get("turns", [])
    for idx in range(1, min(len(chosen_turns), len(rejected_turns))):
        if repair_state_signature(chosen, idx) != repair_state_signature(rejected, idx):
            skipped += 1
            continue
        chosen_plan = chosen_turns[idx].get("plan", "")
        rejected_plan = rejected_turns[idx].get("plan", "")
        if not chosen_plan or not rejected_plan or chosen_plan == rejected_plan:
            skipped += 1
            continue
        conversations = [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": build_repair_prompt(chosen, idx)},
        ]
        item = to_sharegpt(conversations, chosen_plan, rejected_plan)
        item["metadata"] = {
            "stage": "leader_repair_strict",
            "round": idx + 1,
            "task_id": pair.get("task_id"),
            "chosen_trajectory_id": pair.get("chosen_trajectory_id"),
            "rejected_trajectory_id": pair.get("rejected_trajectory_id"),
            "utility_delta": pair.get("utility_delta"),
            "chosen_utility": pair.get("chosen_utility"),
            "rejected_utility": pair.get("rejected_utility"),
            "weight": pair.get("weight", 1.0),
        }
        examples.append(item)
    return examples, skipped


def strip_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k != "metadata"}


def register_dataset(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    with dataset_info_path.open("r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "ranking": True,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    }
    backup_path = dataset_info_path.with_suffix(dataset_info_path.suffix + ".bak_leader_round")
    if not backup_path.exists():
        backup_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with dataset_info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        f.write("\n")


def convert(args: argparse.Namespace, strong_only: bool, dataset_name: str) -> dict[str, Any]:
    preferences = load_jsonl(Path(args.preferences))
    trajectories = {row["trajectory_id"]: row for row in load_jsonl(Path(args.trajectories))}
    examples: list[dict[str, Any]] = []
    missing_refs = 0
    skipped = 0
    repair_strict_skipped = 0

    for pair in preferences:
        if strong_only:
            chosen_passed = bool(pair.get("chosen_metrics", {}).get("passed"))
            rejected_passed = bool(pair.get("rejected_metrics", {}).get("passed"))
            if not (chosen_passed and not rejected_passed):
                skipped += 1
                continue
        chosen = trajectories.get(pair.get("chosen_trajectory_id"))
        rejected = trajectories.get(pair.get("rejected_trajectory_id"))
        if chosen is None or rejected is None:
            missing_refs += 1
            continue
        initial = make_initial_example(pair, chosen, rejected)
        if initial is None:
            skipped += 1
        else:
            examples.append(initial)
        if args.include_repair_strict:
            repairs, skipped_repairs = make_repair_examples_strict(pair, chosen, rejected)
            examples.extend(repairs)
            repair_strict_skipped += skipped_repairs

    data_dir = Path(args.llamafactory_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dpo_path = data_dir / f"{dataset_name}.json"
    with dpo_path.open("w", encoding="utf-8") as f:
        json.dump([strip_metadata(item) for item in examples], f, ensure_ascii=False, indent=2)
        f.write("\n")

    wpo_dir = Path(args.output_dir)
    wpo_dir.mkdir(parents=True, exist_ok=True)
    wpo_path = wpo_dir / f"{dataset_name}_wpo.jsonl"
    with wpo_path.open("w", encoding="utf-8") as f:
        for item in examples:
            payload = dict(item)
            payload["weight"] = float(item.get("metadata", {}).get("weight", 1.0))
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    register_dataset(Path(args.dataset_info), dataset_name, dpo_path.name)
    stages: dict[str, int] = {}
    for item in examples:
        stage = item.get("metadata", {}).get("stage", "unknown")
        stages[stage] = stages.get(stage, 0) + 1
    return {
        "dataset_name": dataset_name,
        "strong_only": strong_only,
        "num_input_preferences": len(preferences),
        "num_output_examples": len(examples),
        "stages": stages,
        "num_skipped": skipped,
        "num_missing_refs": missing_refs,
        "num_repair_strict_skipped": repair_strict_skipped,
        "dpo_path": str(dpo_path),
        "wpo_path": str(wpo_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert leader trajectory pairs into round-level planner DPO/WPO data.")
    parser.add_argument("--preferences", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--llamafactory-data-dir", default="/workspace/LLaMA-Factory/data")
    parser.add_argument("--dataset-info", default="/workspace/LLaMA-Factory/data/dataset_info.json")
    parser.add_argument("--output-dir", default="/workspace/multi_agent/stackelberg_codepo/outputs/preference_demo")
    parser.add_argument("--all-name", default="leader_planner_round_dpo_all")
    parser.add_argument("--strong-name", default="leader_planner_round_dpo_strong")
    parser.add_argument("--include-repair-strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [
        convert(args, strong_only=True, dataset_name=args.strong_name),
        convert(args, strong_only=False, dataset_name=args.all_name),
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

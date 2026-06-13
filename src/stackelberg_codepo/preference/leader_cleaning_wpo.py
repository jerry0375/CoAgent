#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


OVERREACH_PATTERNS = [
    re.compile(r"```", re.IGNORECASE),
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\(", re.IGNORECASE),
    re.compile(r"\bassert\s+", re.IGNORECASE),
    re.compile(r"\btest\s+cases?\b", re.IGNORECASE),
    re.compile(r"\bpython\s+code\s*:", re.IGNORECASE),
    re.compile(r"\bpython\s+function\b", re.IGNORECASE),
    re.compile(r"\b(?:here(?:'s| is)?|below is)\s+(?:the\s+)?(?:corrected\s+|final\s+)?(?:implementation|code|function)\b", re.IGNORECASE),
    re.compile(r"\b(?:revised|corrected|final)\s+code\s*:", re.IGNORECASE),
    re.compile(r"^\s*(?:implementation|code)\s*:", re.IGNORECASE | re.MULTILINE),
]


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


def response_text(item: dict[str, Any], key: str) -> str:
    value = item[key]
    if isinstance(value, dict):
        return str(value.get("value", ""))
    return str(value)


def set_response_text(item: dict[str, Any], key: str, text: str) -> None:
    value = item[key]
    text = text.rstrip() + "\n"
    if isinstance(value, dict):
        value["value"] = text
    else:
        item[key] = text


def strip_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k != "metadata"}


def task_id(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if metadata.get("task_id"):
        return str(metadata["task_id"])
    human = ""
    for message in item.get("conversations", []):
        if message.get("from") in {"human", "user"}:
            human = str(message.get("value", ""))
            break
    match = re.search(r"Task ID:\s*(\S+)", human)
    return match.group(1) if match else "unknown"


def utility_delta(item: dict[str, Any]) -> float:
    metadata = item.get("metadata", {})
    value = metadata.get("utility_delta", item.get("utility_delta", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def trajectory_id(item: dict[str, Any], key: str) -> str | None:
    metadata = item.get("metadata", {})
    value = metadata.get(key, item.get(key))
    return str(value) if value else None


def trajectory_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": bool(row.get("passed", False)),
        "pass_rate": float(row.get("pass_rate", row.get("pass_rate_raw", 0.0)) or 0.0),
        "leader_utility": row.get("leader_utility"),
        "role_boundary_penalty_total": row.get("role_boundary_penalty_total"),
    }


def load_trajectory_metrics(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = load_jsonl(Path(path))
    metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        tid = row.get("trajectory_id") or row.get("id")
        if tid:
            metrics[str(tid)] = trajectory_metrics(row)
    return metrics


def metadata_float(item: dict[str, Any], key: str, default: float = 0.0) -> float:
    metadata = item.get("metadata", {})
    value = metadata.get(key, item.get(key, default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def has_overreach(text: str) -> bool:
    return any(pattern.search(text) for pattern in OVERREACH_PATTERNS)


def normalize_for_dedup(text: str) -> str:
    text = re.sub(r"^\s*estimated\s+effort\s+tokens\s*:\s*\d+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)

    def pick(q: float) -> float:
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return float(values[idx])

    return {
        "min": float(values[0]),
        "p25": pick(0.25),
        "p50": float(median(values)),
        "mean": float(mean(values)),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": float(values[-1]),
    }


def register_dataset(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_info_path.exists():
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        backup_path = dataset_info_path.with_suffix(dataset_info_path.suffix + ".bak_clean_initial_leader")
        if not backup_path.exists():
            backup_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        dataset_info = {}
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
    dataset_info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean sanitized initial leader WPO pairs for planner-only DPO/WPO training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-wpo", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--dpo-output", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-info", default="/workspace/LLaMA-Factory/data/dataset_info.json")
    parser.add_argument("--min-response-chars", type=int, default=80)
    parser.add_argument("--max-pairs-per-task", type=int, default=3)
    parser.add_argument("--drop-rejected-overreach", action="store_true")
    parser.add_argument("--trajectories", default=None, help="Optional trajectory JSONL used for chosen pass-rate quality gates.")
    parser.add_argument("--min-chosen-pass-rate", type=float, default=None)
    parser.add_argument("--require-chosen-passed", action="store_true")
    parser.add_argument("--min-chosen-utility", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_jsonl(Path(args.input))
    trajectory_metrics_by_id = load_trajectory_metrics(args.trajectories)
    stats: dict[str, Any] = {
        "input": args.input,
        "num_input": len(rows),
        "trajectories": args.trajectories,
        "num_trajectory_metrics": len(trajectory_metrics_by_id),
        "min_response_chars": args.min_response_chars,
        "max_pairs_per_task": args.max_pairs_per_task,
        "drop_rejected_overreach": bool(args.drop_rejected_overreach),
        "min_chosen_pass_rate": args.min_chosen_pass_rate,
        "require_chosen_passed": bool(args.require_chosen_passed),
        "min_chosen_utility": args.min_chosen_utility,
        "dropped_short": 0,
        "dropped_chosen_overreach": 0,
        "dropped_rejected_overreach": 0,
        "dropped_missing_chosen_metrics": 0,
        "dropped_chosen_pass_rate": 0,
        "dropped_chosen_not_passed": 0,
        "dropped_chosen_utility": 0,
        "dropped_duplicate_chosen": 0,
        "dropped_task_cap": 0,
    }

    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = json.loads(json.dumps(row, ensure_ascii=False))
        chosen = response_text(item, "chosen").strip()
        rejected = response_text(item, "rejected").strip()
        if len(chosen) < args.min_response_chars or len(rejected) < args.min_response_chars:
            stats["dropped_short"] += 1
            continue
        if has_overreach(chosen):
            stats["dropped_chosen_overreach"] += 1
            continue
        if args.drop_rejected_overreach and has_overreach(rejected):
            stats["dropped_rejected_overreach"] += 1
            continue
        chosen_tid = trajectory_id(item, "chosen_trajectory_id")
        chosen_metrics = trajectory_metrics_by_id.get(chosen_tid or "")
        if args.min_chosen_pass_rate is not None or args.require_chosen_passed:
            if not chosen_metrics:
                stats["dropped_missing_chosen_metrics"] += 1
                continue
            chosen_pass_rate = float(chosen_metrics.get("pass_rate", 0.0))
            if args.min_chosen_pass_rate is not None and chosen_pass_rate < args.min_chosen_pass_rate:
                stats["dropped_chosen_pass_rate"] += 1
                continue
            if args.require_chosen_passed and not bool(chosen_metrics.get("passed", False)):
                stats["dropped_chosen_not_passed"] += 1
                continue
        if args.min_chosen_utility is not None and metadata_float(item, "chosen_utility") < args.min_chosen_utility:
            stats["dropped_chosen_utility"] += 1
            continue
        set_response_text(item, "chosen", chosen)
        set_response_text(item, "rejected", rejected)
        item.setdefault("metadata", {})
        if chosen_metrics:
            item["metadata"]["chosen_passed"] = bool(chosen_metrics.get("passed", False))
            item["metadata"]["chosen_pass_rate"] = float(chosen_metrics.get("pass_rate", 0.0))
            item["metadata"]["chosen_role_boundary_penalty_total"] = chosen_metrics.get("role_boundary_penalty_total")
        item["metadata"]["clean_initial_leader_text"] = True
        item["metadata"]["clean_initial_source"] = args.input
        candidates.append(item)

    candidates.sort(key=lambda item: (task_id(item), -utility_delta(item), -float(item.get("weight", 1.0))))

    seen_chosen: set[str] = set()
    task_counts: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for item in candidates:
        chosen_key = text_hash(response_text(item, "chosen"))
        if chosen_key in seen_chosen:
            stats["dropped_duplicate_chosen"] += 1
            continue
        tid = task_id(item)
        if args.max_pairs_per_task > 0 and task_counts[tid] >= args.max_pairs_per_task:
            stats["dropped_task_cap"] += 1
            continue
        seen_chosen.add(chosen_key)
        task_counts[tid] += 1
        output_rows.append(item)

    output_wpo = Path(args.output_wpo)
    output_wpo.parent.mkdir(parents=True, exist_ok=True)
    with output_wpo.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if args.dpo_output:
        dpo_path = Path(args.dpo_output)
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        dpo_path.write_text(json.dumps([strip_metadata(item) for item in output_rows], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stats["dpo_output"] = str(dpo_path)
        if args.dataset_name:
            register_dataset(Path(args.dataset_info), args.dataset_name, dpo_path.name)
            stats["dataset_name"] = args.dataset_name

    weights = [float(item.get("weight", 1.0)) for item in output_rows]
    stats.update(
        {
            "num_candidates_after_content_filters": len(candidates),
            "num_output": len(output_rows),
            "output_wpo": str(output_wpo),
            "task_counts": dict(sorted(task_counts.items())),
            "weight_stats": percentiles(weights),
        }
    )

    summary_path = Path(args.summary) if args.summary else output_wpo.with_suffix(output_wpo.suffix + ".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stats["summary"] = str(summary_path)
    summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

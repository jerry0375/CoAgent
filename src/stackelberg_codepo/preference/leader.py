from __future__ import annotations

import itertools
from typing import Any

from stackelberg_codepo.preference.utility import leader_utility, weight_from_delta
from stackelberg_codepo.schemas import UtilityConfig


def enrich_trajectory(row: dict[str, Any], pass_rate: float, cfg: UtilityConfig) -> dict[str, Any]:
    total_tokens = int(row.get("planner_tokens", 0)) + int(row.get("coder_tokens", 0))
    utility = leader_utility(pass_rate, int(row.get("rounds", 1)), total_tokens, cfg)
    return {
        **row,
        "pass_rate": pass_rate,
        "total_tokens": total_tokens,
        **utility,
    }


def build_leader_preferences(trajectories: list[dict[str, Any]], cfg: UtilityConfig) -> list[dict[str, Any]]:
    raw_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    by_task: dict[str, list[dict[str, Any]]] = {}
    for traj in trajectories:
        by_task.setdefault(str(traj["task_id"]), []).append(traj)

    for task_rows in by_task.values():
        for left, right in itertools.combinations(task_rows, 2):
            diff = float(left["leader_utility"]) - float(right["leader_utility"])
            if abs(diff) <= cfg.leader_margin:
                continue
            chosen, rejected = (left, right) if diff > 0 else (right, left)
            raw_pairs.append((chosen, rejected, abs(diff)))

    if not raw_pairs:
        return []
    avg_delta = sum(delta for _, _, delta in raw_pairs) / len(raw_pairs)
    pairs: list[dict[str, Any]] = []
    for chosen, rejected, delta in raw_pairs:
        weight = weight_from_delta(delta, avg_delta, cfg)
        pairs.append(
            {
                "task_id": chosen["task_id"],
                "stage": "leader_initial",
                "chosen": chosen.get("plan", ""),
                "rejected": rejected.get("plan", ""),
                "chosen_trajectory_id": chosen.get("trajectory_id"),
                "rejected_trajectory_id": rejected.get("trajectory_id"),
                "chosen_utility": chosen["leader_utility"],
                "rejected_utility": rejected["leader_utility"],
                "utility_delta": delta,
                "weight": weight,
            }
        )
    return pairs


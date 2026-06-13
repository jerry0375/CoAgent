from __future__ import annotations

import itertools
from typing import Any

from stackelberg_codepo.preference.utility import follower_utility, weight_from_delta
from stackelberg_codepo.schemas import UtilityConfig


def build_follower_preferences(candidates: list[dict[str, Any]], cfg: UtilityConfig) -> list[dict[str, Any]]:
    enriched = []
    for row in candidates:
        output_tokens = int(row.get("coder_tokens", row.get("output_tokens", 0)))
        utility = follower_utility(float(row.get("pass_rate", 0.0)), output_tokens, cfg, float(row.get("incentive", 0.0)))
        enriched.append({**row, **utility})

    raw_pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    by_state: dict[str, list[dict[str, Any]]] = {}
    for item in enriched:
        by_state.setdefault(str(item["state_id"]), []).append(item)
    for state_rows in by_state.values():
        for left, right in itertools.combinations(state_rows, 2):
            diff = float(left["follower_utility"]) - float(right["follower_utility"])
            if abs(diff) <= cfg.follower_margin:
                continue
            chosen, rejected = (left, right) if diff > 0 else (right, left)
            raw_pairs.append((chosen, rejected, abs(diff)))

    if not raw_pairs:
        return []
    avg_delta = sum(delta for _, _, delta in raw_pairs) / len(raw_pairs)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chosen, rejected, delta in raw_pairs:
        key = (str(chosen["state_id"]), str(chosen.get("code", "")).strip(), str(rejected.get("code", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "task_id": chosen["task_id"],
                "state_id": chosen["state_id"],
                "stage": "follower_code",
                "prompt": chosen.get("prompt", ""),
                "chosen": chosen.get("code", ""),
                "rejected": rejected.get("code", ""),
                "chosen_utility": chosen["follower_utility"],
                "rejected_utility": rejected["follower_utility"],
                "utility_delta": delta,
                "weight": weight_from_delta(delta, avg_delta, cfg),
            }
        )
    return pairs

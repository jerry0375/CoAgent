from __future__ import annotations

from typing import Any


LEADER_SYSTEM = (
    "You are the leader/planner in a planner-coder code generation team. "
    "Give concise, actionable guidance. Do not write code."
)

FOLLOWER_SYSTEM = (
    "You are the follower/coder in a planner-coder code generation team. "
    "Return only valid Python code. Do not include markdown fences, prose, tests, or examples."
)


def _sharegpt(system: str, human: str, chosen: str, rejected: str, metadata: dict[str, Any], weight: float) -> dict[str, Any]:
    return {
        "conversations": [
            {"from": "system", "value": system},
            {"from": "human", "value": human},
        ],
        "chosen": {"from": "gpt", "value": chosen.strip() + "\n"},
        "rejected": {"from": "gpt", "value": rejected.strip() + "\n"},
        "metadata": metadata,
        "weight": float(weight),
    }


def leader_pair_to_wpo(pair: dict[str, Any], trajectories_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    chosen_traj = trajectories_by_id.get(str(pair.get("chosen_trajectory_id")), {})
    raw = chosen_traj.get("raw", chosen_traj)
    state_text = raw.get("state_text", "")
    human = (
        f"Task state:\n{state_text}\n\n"
        "Create an implementation plan for the coder.\n"
        "Start with exactly one line: Estimated effort tokens: N\n"
        "Then mention the core algorithm and key edge cases. Do not write code."
    )
    metadata = {
        "stage": pair.get("stage", "leader_initial"),
        "task_id": pair.get("task_id"),
        "chosen_trajectory_id": pair.get("chosen_trajectory_id"),
        "rejected_trajectory_id": pair.get("rejected_trajectory_id"),
        "chosen_utility": pair.get("chosen_utility"),
        "rejected_utility": pair.get("rejected_utility"),
        "utility_delta": pair.get("utility_delta"),
        "weight": pair.get("weight", 1.0),
    }
    return _sharegpt(LEADER_SYSTEM, human, pair.get("chosen", ""), pair.get("rejected", ""), metadata, float(pair.get("weight", 1.0)))


def follower_pair_to_wpo(pair: dict[str, Any]) -> dict[str, Any]:
    human = pair.get("prompt") or (
        f"Task state id: {pair.get('state_id')}\n\n"
        "Write the complete Python function requested by the task. Return Python code only."
    )
    metadata = {
        "stage": pair.get("stage", "follower_code"),
        "task_id": pair.get("task_id"),
        "state_id": pair.get("state_id"),
        "chosen_utility": pair.get("chosen_utility"),
        "rejected_utility": pair.get("rejected_utility"),
        "utility_delta": pair.get("utility_delta"),
        "weight": pair.get("weight", 1.0),
    }
    return _sharegpt(FOLLOWER_SYSTEM, human, pair.get("chosen", ""), pair.get("rejected", ""), metadata, float(pair.get("weight", 1.0)))

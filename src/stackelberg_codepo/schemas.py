from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    entry_point: str
    prompt: str
    tests: list[dict[str, Any]]


@dataclass(frozen=True)
class UtilityConfig:
    lambda_round: float
    lambda_token: float
    leader_margin: float
    follower_margin: float
    weight_min: float
    weight_max: float
    weight_power: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UtilityConfig":
        return cls(
            lambda_round=float(data.get("lambda_round", 0.02)),
            lambda_token=float(data.get("lambda_token", 1e-5)),
            leader_margin=float(data.get("leader_margin", 0.03)),
            follower_margin=float(data.get("follower_margin", 0.02)),
            weight_min=float(data.get("weight_min", 0.3)),
            weight_max=float(data.get("weight_max", 2.0)),
            weight_power=float(data.get("weight_power", 0.65)),
        )


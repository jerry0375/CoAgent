from __future__ import annotations

from stackelberg_codepo.schemas import UtilityConfig


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def leader_utility(pass_rate: float, rounds: int, total_tokens: int, cfg: UtilityConfig, incentive: float = 0.0, role_penalty: float = 0.0) -> dict[str, float]:
    quality = clamp01(pass_rate)
    round_cost = cfg.lambda_round * float(rounds)
    token_cost = cfg.lambda_token * float(total_tokens)
    total_cost = round_cost + token_cost
    utility = quality - total_cost - float(incentive) - float(role_penalty)
    return {
        "quality": quality,
        "round_cost": round_cost,
        "token_cost": token_cost,
        "total_cost": total_cost,
        "leader_utility": utility,
    }


def follower_utility(pass_rate: float, output_tokens: int, cfg: UtilityConfig, incentive: float = 0.0) -> dict[str, float]:
    quality = clamp01(pass_rate)
    token_cost = cfg.lambda_token * float(output_tokens)
    utility = quality + float(incentive) - token_cost
    return {
        "quality": quality,
        "token_cost": token_cost,
        "follower_utility": utility,
    }


def weight_from_delta(delta: float, average_delta: float, cfg: UtilityConfig) -> float:
    raw = float(delta) / (float(average_delta) + 1e-8)
    compressed = raw ** cfg.weight_power
    return clamp(compressed, cfg.weight_min, cfg.weight_max)


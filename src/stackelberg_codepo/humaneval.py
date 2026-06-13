from __future__ import annotations

import ast
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Literal

from stackelberg_codepo.config import ExperimentConfig as Config

Decision = Literal["continue", "stop"]
Difficulty = Literal["easy", "medium", "hard"]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class Task:
    task_id: str
    source_task_id: str
    prompt: str
    test: str
    entry_point: str
    split: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionFeedback:
    task_id: str
    passed: bool
    pass_rate: float
    passed_tests: int
    failed_tests: int
    total_tests: int
    execution_time: float
    memory_mb: float | None
    timeout: bool
    error_type: str | None
    error_message: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionState:
    task: Task
    history_summary: str = ""
    previous_code: str = ""
    previous_feedback: ExecutionFeedback | None = None
    round_index: int = 1


@dataclass(frozen=True)
class PlannerAction:
    instruction: str
    decision: Decision
    incentive_rule: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeResponse:
    code: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    state: InteractionState
    planner_action: PlannerAction
    code_response: CodeResponse | None
    feedback: ExecutionFeedback | None
    incentive: float = 0.0
    prompt_tokens: int = 0
    code_tokens: int = 0


@dataclass(frozen=True)
class Trajectory:
    task: Task
    turns: list[Turn]

    @property
    def final_feedback(self) -> ExecutionFeedback | None:
        for turn in reversed(self.turns):
            if turn.feedback is not None:
                return turn.feedback
        return None

    @property
    def rounds(self) -> int:
        return len(self.turns)

    @property
    def prompt_tokens(self) -> int:
        return sum(turn.prompt_tokens for turn in self.turns)

    @property
    def code_tokens(self) -> int:
        return sum(turn.code_tokens for turn in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.code_tokens

    @property
    def total_incentive(self) -> float:
        return sum(turn.incentive for turn in self.turns)


def read_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc


def load_tasks(path: str | Path, limit: int | None = None) -> list[Task]:
    tasks: list[Task] = []
    for raw in read_jsonl(path):
        tasks.append(
            Task(
                task_id=str(raw["task_id"]),
                source_task_id=str(raw.get("source_task_id", raw["task_id"])),
                prompt=str(raw["prompt"]),
                test=str(raw["test"]),
                entry_point=str(raw["entry_point"]),
                split=str(raw.get("split", "")),
                raw=raw,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


class HumanEvalExecutor:
    def __init__(self, phase1_dir: str | Path):
        module_path = Path(phase1_dir).resolve() / "execute_code.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Cannot find Phase 1 executor: {module_path}")
        spec = importlib.util.spec_from_file_location("humaneval_phase1_execute_code", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import executor from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module = module

    def execute(self, task: Task, code: str, code_type: str) -> ExecutionFeedback:
        raw = self._module.execute_task(task.raw, code, code_type)
        return ExecutionFeedback(
            task_id=str(raw.get("task_id", "")),
            passed=bool(raw.get("passed", False)),
            pass_rate=clamp01(float(raw.get("pass_rate", 0.0))),
            passed_tests=int(raw.get("passed_tests", 0)),
            failed_tests=int(raw.get("failed_tests", 0)),
            total_tests=int(raw.get("total_tests", 0)),
            execution_time=float(raw.get("execution_time", 0.0)),
            memory_mb=None if raw.get("memory_mb") is None else float(raw["memory_mb"]),
            timeout=bool(raw.get("timeout", False)),
            error_type=raw.get("error_type"),
            error_message=raw.get("error_message"),
            raw=raw,
        )


class TokenCounter:
    def __init__(self, approx_chars_per_token: float):
        if approx_chars_per_token <= 0:
            raise ValueError("approx_chars_per_token must be positive")
        self.approx_chars_per_token = approx_chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return int(math.ceil(len(text) / self.approx_chars_per_token))


def count_asserts(test_code: str) -> int:
    try:
        tree = ast.parse(test_code or "")
    except SyntaxError:
        return 0
    return sum(isinstance(node, ast.Assert) for node in ast.walk(tree))


def classify_difficulty(task: Task, cfg: Config) -> Difficulty:
    diff = cfg.table("difficulty")
    n_asserts = count_asserts(task.test)
    if n_asserts <= int(diff["easy_max_asserts"]):
        return "easy"
    if n_asserts <= int(diff["medium_max_asserts"]):
        return "medium"
    return "hard"


def total_cost(trajectory: Trajectory, cfg: Config) -> float:
    cost = cfg.table("cost")
    return (
        float(cost["lambda_round"]) * trajectory.rounds
        + float(cost["lambda_token"]) * trajectory.total_tokens
    )


def efficiency_score(feedback: ExecutionFeedback | None) -> float:
    if feedback is None or feedback.timeout:
        return 0.0
    return 1.0 / (1.0 + max(0.0, feedback.execution_time))


def quality_score(feedback: ExecutionFeedback | None, cfg: Config) -> float:
    if feedback is None:
        return 0.0
    # TODO(efficiency): when the executor exposes per-test time and memory
    # limits, use them as LeetCode-style constraints for each testcase. Do not
    # add efficiency as a bonus to Q here; Q should remain a clipped [0, 1]
    # constrained pass ratio so utility magnitudes stay interpretable.
    return clamp01(feedback.pass_rate)


def leader_utility(trajectory: Trajectory, cfg: Config) -> float:
    return quality_score(trajectory.final_feedback, cfg) - total_cost(trajectory, cfg) - trajectory.total_incentive


def follower_utility(
    feedback: ExecutionFeedback | None,
    previous_pass_rate: float,
    code_tokens: int,
    incentive: float,
    cfg: Config,
) -> float:
    cost = cfg.table("cost")
    if feedback is None:
        reward = 0.0
    else:
        delta = max(0.0, feedback.pass_rate - previous_pass_rate)
        reward = float(cost["beta_pass"]) * feedback.pass_rate + float(cost["beta_delta_pass"]) * delta
    return reward - float(cost["lambda_follower_token"]) * code_tokens + incentive

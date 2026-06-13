from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingJob:
    role: str
    data_path: Path
    output_dir: Path
    max_steps: int = 10
    learning_rate: float = 5e-6


def build_weighted_dpo_command(job: TrainingJob, script_path: Path, model_path: Path, device: str = "cuda:0") -> list[str]:
    return [
        "/opt/conda/bin/python",
        str(script_path),
        "--model-path",
        str(model_path),
        "--data",
        str(job.data_path),
        "--output-dir",
        str(job.output_dir),
        "--device",
        device,
        "--max-steps",
        str(job.max_steps),
        "--learning-rate",
        str(job.learning_rate),
        "--normalize-logprob",
    ]


def adapter_exists(path: str | Path) -> bool:
    adapter_dir = Path(path)
    return (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()


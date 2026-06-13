from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(config_path.resolve().parents[1])
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_project_root"]) / path


def table(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be an object")
    return value



class ExperimentConfig:
    """Config adapter for migrated experiment code.

    The refactored project uses JSON configs, while the original validated demo
    used TOML. This adapter supports both so migrated modules do not need to
    depend on the previous project package.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.root = self.path.parent.parent
        text = self.path.read_text(encoding="utf-8")
        if self.path.suffix.lower() == ".json":
            self.raw = json.loads(text)
        else:
            self.raw = _load_toml_like(self.path)

    def table(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise KeyError(f"Missing config table: {name}")
        value = self.raw[name]
        if not isinstance(value, dict):
            raise TypeError(f"Config section {name!r} must be an object")
        return value

    def methods(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("methods", {})

    def enabled_methods(self) -> dict[str, dict[str, Any]]:
        return {k: v for k, v in self.methods().items() if bool(v.get("enabled", True))}

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    @property
    def phase1_dir(self) -> Path:
        return self.resolve(str(self.table("paths")["phase1_dir"]))

    @property
    def task_file(self) -> Path:
        return self.resolve(str(self.table("paths")["task_file"]))

    @property
    def train_file(self) -> Path:
        return self.resolve(str(self.table("paths")["train_file"]))

    @property
    def valid_file(self) -> Path:
        return self.resolve(str(self.table("paths")["valid_file"]))


def _load_toml_like(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore
        with path.open("rb") as f:
            return tomllib.load(f)
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore
            with path.open("rb") as f:
                return tomli.load(f)
        except ModuleNotFoundError:
            pass
    # Small fallback for the simple TOML used by the old configs.
    root: dict[str, Any] = {}
    current = root
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = root
            for part in line[1:-1].split("."):
                current = current.setdefault(part, {})
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = _parse_config_scalar(value.strip())
    return root


def _parse_config_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_config_scalar(part.strip()) for part in inner.split(",")]
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value

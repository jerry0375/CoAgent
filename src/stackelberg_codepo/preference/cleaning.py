from __future__ import annotations

import re
from typing import Any


OVERREACH_PATTERNS = [
    re.compile(r"```", re.IGNORECASE),
    re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\(", re.IGNORECASE),
    re.compile(r"\bassert\s+", re.IGNORECASE),
    re.compile(r"\b(?:here(?:'s| is)?|below is)\s+(?:the\s+)?(?:implementation|code|function)\b", re.IGNORECASE),
]


def has_planner_overreach(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in OVERREACH_PATTERNS)


def clean_leader_preferences(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    stats = {"input": len(rows), "dropped_overreach": 0, "dropped_duplicate": 0, "output": 0}
    for row in rows:
        chosen = str(row.get("chosen", ""))
        if has_planner_overreach(chosen):
            stats["dropped_overreach"] += 1
            continue
        key = re.sub(r"\s+", " ", chosen.strip().lower())
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        cleaned.append(row)
    stats["output"] = len(cleaned)
    return cleaned, stats


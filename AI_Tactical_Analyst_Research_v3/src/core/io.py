"""Filesystem helpers and serialization utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import CSV_EXPORTS, HEATMAP_DIR, OUTPUT_DIR, REPORT_DIR


def ensure_project_dirs() -> None:
    """Create all runtime output directories if missing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def to_serializable(value: Any) -> Any:
    """Convert custom objects to JSON-serializable values."""
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with UTF-8 encoding and indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a Power BI compatible UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def expected_output_paths() -> list[Path]:
    """List expected primary output files."""
    return [OUTPUT_DIR / name for name in CSV_EXPORTS]

"""Lightweight smoke tests for repository integrity."""

from __future__ import annotations

from pathlib import Path

from src.core.config import ConfigManager
from src.visualization.reports import ReportGenerator
from src.core.logging_utils import configure_logging


def test_config_loads() -> None:
    config = ConfigManager.load(Path("configs/default.yaml"))
    assert config.detector.confidence > 0
    assert config.tracker.max_time_lost >= 1


def test_powerbi_guide_generation(tmp_path: Path) -> None:
    logger = configure_logging(tmp_path / "test.log")
    path = ReportGenerator(logger).generate_powerbi_guide(tmp_path / "guide.pdf")
    assert path.exists()
    assert path.stat().st_size > 1000

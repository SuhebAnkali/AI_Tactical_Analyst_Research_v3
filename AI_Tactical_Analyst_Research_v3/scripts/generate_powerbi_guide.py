"""Generate the Power BI PDF guide independently."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.logging_utils import configure_logging
from src.visualization.reports import ReportGenerator

logger = configure_logging(Path("outputs/powerbi_guide.log"))
path = ReportGenerator(logger).generate_powerbi_guide()
print(path)

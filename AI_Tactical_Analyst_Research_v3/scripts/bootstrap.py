"""Runtime dependency bootstrapper."""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Iterable

PACKAGE_MAP = {
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "shapely": "shapely",
    "supervision": "supervision",
    "torch": "torch",
    "ultralytics": "ultralytics",
    "reportlab": "reportlab",
}


def ensure_runtime_dependencies(extra_modules: Iterable[str] | None = None) -> None:
    """Install missing Python dependencies at runtime if required."""
    modules = set(PACKAGE_MAP.keys())
    if extra_modules:
        modules.update(extra_modules)
    missing_packages: list[str] = []
    for module_name in sorted(modules):
        try:
            importlib.import_module(module_name)
        except Exception:
            package_name = PACKAGE_MAP.get(module_name, module_name)
            missing_packages.append(package_name)
    if missing_packages:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "--user",
            "--upgrade-strategy",
            "only-if-needed",
            *sorted(set(missing_packages)),
        ])


if __name__ == "__main__":
    ensure_runtime_dependencies()
    print("Runtime dependencies verified.")

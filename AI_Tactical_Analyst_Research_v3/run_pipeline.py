"""CLI entry point for tactical video analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.bootstrap import ensure_runtime_dependencies

ensure_runtime_dependencies()

from src.core.config import ConfigManager
from src.core.logging_utils import configure_logging
from src.core.pipeline import TacticalAnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Tactical Analyst Research v3")
    parser.add_argument("--video", required=True, help="Input football video path")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration YAML path")
    parser.add_argument("--model", default=None, help="Override YOLO model path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for debugging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = ConfigManager.load(args.config)
    if args.model:
        config.detector.model_path = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_frames is not None:
        config.max_frames = args.max_frames
    logger = configure_logging(Path(config.output_dir) / "pipeline.log")
    pipeline = TacticalAnalysisPipeline(config, logger)
    outputs = pipeline.run(args.video)
    logger.info("Generated outputs:")
    for name, path in outputs.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()

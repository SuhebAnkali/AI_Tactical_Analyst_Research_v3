"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import CONFIG_DIR, MODEL_DIR, OUTPUT_DIR


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = str(MODEL_DIR / "yolo11n.pt")
    confidence: float = 0.25
    iou: float = 0.45
    image_size: int = 1280
    device: str = "cpu"
    half_precision: bool = False
    ball_confidence: float = 0.15


@dataclass(slots=True)
class TrackerConfig:
    high_thresh: float = 0.35
    low_thresh: float = 0.10
    new_track_thresh: float = 0.45
    match_iou_thresh: float = 0.30
    ball_match_iou_thresh: float = 0.15
    max_time_lost: int = 45
    min_hits: int = 2


@dataclass(slots=True)
class TeamClassifierConfig:
    shirt_crop_top_ratio: float = 0.18
    shirt_crop_bottom_ratio: float = 0.58
    kmeans_clusters: int = 3
    temporal_window: int = 30
    min_pixels: int = 120
    color_history_decay: float = 0.85
    grass_hsv_low: tuple[int, int, int] = (28, 25, 25)
    grass_hsv_high: tuple[int, int, int] = (95, 255, 255)


@dataclass(slots=True)
class HomographyConfig:
    auto_calibrate: bool = True
    manual_calibration_path: str = str(CONFIG_DIR / "calibration_sample.json")
    refresh_interval_frames: int = 90
    line_threshold: int = 120
    min_intersections: int = 4
    ransac_reproj_threshold: float = 5.0


@dataclass(slots=True)
class FormationConfig:
    update_interval_frames: int = 10
    smoothing_window: int = 24
    min_players: int = 7
    direction_flip_margin: float = 3.5
    clustering_bandwidth: float = 9.0


@dataclass(slots=True)
class HeatmapConfig:
    bins_x: int = 64
    bins_y: int = 42
    gaussian_sigma: float = 1.5


@dataclass(slots=True)
class EventConfig:
    possession_radius_m: float = 2.3
    pass_min_distance_m: float = 4.0
    shot_min_speed_mps: float = 8.0
    shot_final_third_x_m: float = 78.0
    ball_recovery_gap_frames: int = 12
    touch_proximity_px: float = 55.0
    trail_length: int = 25


@dataclass(slots=True)
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    team_classifier: TeamClassifierConfig = field(default_factory=TeamClassifierConfig)
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    formation: FormationConfig = field(default_factory=FormationConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    events: EventConfig = field(default_factory=EventConfig)
    output_dir: str = str(OUTPUT_DIR)
    export_annotated_video: bool = True
    annotate_video: bool = True
    video_stride: int = 1
    max_frames: int | None = None


class ConfigManager:
    """Load YAML configuration files into typed dataclasses."""

    @staticmethod
    def load(path: str | Path | None = None) -> AppConfig:
        config_path = Path(path) if path else CONFIG_DIR / "default.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return ConfigManager.from_dict(payload)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> AppConfig:
        return AppConfig(
            detector=DetectorConfig(**payload.get("detector", {})),
            tracker=TrackerConfig(**payload.get("tracker", {})),
            team_classifier=TeamClassifierConfig(**payload.get("team_classifier", {})),
            homography=HomographyConfig(**payload.get("homography", {})),
            formation=FormationConfig(**payload.get("formation", {})),
            heatmap=HeatmapConfig(**payload.get("heatmap", {})),
            events=EventConfig(**payload.get("events", {})),
            output_dir=payload.get("output_dir", str(OUTPUT_DIR)),
            export_annotated_video=payload.get("export_annotated_video", True),
            annotate_video=payload.get("annotate_video", True),
            video_stride=payload.get("video_stride", 1),
            max_frames=payload.get("max_frames"),
        )

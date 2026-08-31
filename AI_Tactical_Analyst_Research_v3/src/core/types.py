"""Typed dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Detection:
    """Single frame detection."""

    bbox_xyxy: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    frame_index: int
    model_name: str
    track_id: int | None = None
    team_id: str = "unknown"
    pitch_xy: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0),
            float((self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0),
        )

    @property
    def foot_point(self) -> tuple[float, float]:
        return (float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0), float(self.bbox_xyxy[3]))


@dataclass(slots=True)
class TrackState:
    """State container for tracked objects."""

    track_id: int
    class_id: int
    class_name: str
    bbox_xyxy: np.ndarray
    confidence: float
    frame_index: int
    age: int = 1
    hits: int = 1
    missed: int = 0
    velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    team_id: str = "unknown"
    pitch_xy: tuple[float, float] | None = None
    smoothed_speed_mps: float = 0.0
    history_image_xy: list[tuple[float, float]] = field(default_factory=list)
    history_pitch_xy: list[tuple[float, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def append_position(self, image_xy: tuple[float, float], pitch_xy: tuple[float, float] | None) -> None:
        self.history_image_xy.append(image_xy)
        self.history_image_xy = self.history_image_xy[-300:]
        if pitch_xy is not None:
            self.history_pitch_xy.append(pitch_xy)
            self.history_pitch_xy = self.history_pitch_xy[-300:]


@dataclass(slots=True)
class EventRecord:
    """Structured tactical event."""

    frame_index: int
    event_type: str
    team_id: str
    primary_track_id: int | None
    secondary_track_id: int | None
    ball_track_id: int | None
    start_pitch_xy: tuple[float, float] | None = None
    end_pitch_xy: tuple[float, float] | None = None
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FormationSnapshot:
    """Formation state for one team and one frame."""

    frame_index: int
    team_id: str
    formation: str
    line_counts: list[int]
    confidence: float
    direction: str


@dataclass(slots=True)
class FrameArtifacts:
    """Container returned by each pipeline step for one frame."""

    frame_index: int
    detections: list[Detection]
    tracks: list[TrackState]
    ball_track: TrackState | None
    events: list[EventRecord]
    formations: list[FormationSnapshot]
    annotated_frame: np.ndarray | None = None

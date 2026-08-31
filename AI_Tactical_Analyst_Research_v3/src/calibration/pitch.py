"""Pitch geometry and rendering utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.core.constants import (
    CENTER_CIRCLE_RADIUS_M,
    GOAL_AREA_LENGTH_M,
    GOAL_AREA_WIDTH_M,
    PENALTY_AREA_LENGTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)


@dataclass(slots=True)
class PitchCanvas:
    width_px: int = 900
    height_px: int = 600
    margin_px: int = 30
    background_color: tuple[int, int, int] = (20, 90, 20)
    line_color: tuple[int, int, int] = (240, 240, 240)

    @property
    def scale_x(self) -> float:
        return (self.width_px - 2 * self.margin_px) / PITCH_LENGTH_M

    @property
    def scale_y(self) -> float:
        return (self.height_px - 2 * self.margin_px) / PITCH_WIDTH_M

    def to_canvas(self, pitch_xy: tuple[float, float]) -> tuple[int, int]:
        x, y = pitch_xy
        px = int(round(self.margin_px + x * self.scale_x))
        py = int(round(self.margin_px + y * self.scale_y))
        return px, py

    def blank(self) -> np.ndarray:
        canvas = np.full((self.height_px, self.width_px, 3), self.background_color, dtype=np.uint8)
        self.draw_pitch_lines(canvas)
        return canvas

    def draw_pitch_lines(self, canvas: np.ndarray) -> None:
        tl = self.to_canvas((0.0, 0.0))
        br = self.to_canvas((PITCH_LENGTH_M, PITCH_WIDTH_M))
        cv2.rectangle(canvas, tl, br, self.line_color, 2)
        center_x = self.to_canvas((PITCH_LENGTH_M / 2.0, 0.0))[0]
        cv2.line(canvas, (center_x, tl[1]), (center_x, br[1]), self.line_color, 2)
        center = self.to_canvas((PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0))
        radius = int(round(CENTER_CIRCLE_RADIUS_M * min(self.scale_x, self.scale_y)))
        cv2.circle(canvas, center, radius, self.line_color, 2)
        self._draw_box(canvas, 0.0, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0, PENALTY_AREA_LENGTH_M, PENALTY_AREA_WIDTH_M)
        self._draw_box(canvas, PITCH_LENGTH_M - PENALTY_AREA_LENGTH_M, (PITCH_WIDTH_M - PENALTY_AREA_WIDTH_M) / 2.0, PENALTY_AREA_LENGTH_M, PENALTY_AREA_WIDTH_M)
        self._draw_box(canvas, 0.0, (PITCH_WIDTH_M - GOAL_AREA_WIDTH_M) / 2.0, GOAL_AREA_LENGTH_M, GOAL_AREA_WIDTH_M)
        self._draw_box(canvas, PITCH_LENGTH_M - GOAL_AREA_LENGTH_M, (PITCH_WIDTH_M - GOAL_AREA_WIDTH_M) / 2.0, GOAL_AREA_LENGTH_M, GOAL_AREA_WIDTH_M)

    def _draw_box(self, canvas: np.ndarray, x: float, y: float, w: float, h: float) -> None:
        pt1 = self.to_canvas((x, y))
        pt2 = self.to_canvas((x + w, y + h))
        cv2.rectangle(canvas, pt1, pt2, self.line_color, 2)


PITCH_KEYPOINTS = {
    "top_left": (0.0, 0.0),
    "top_right": (PITCH_LENGTH_M, 0.0),
    "bottom_left": (0.0, PITCH_WIDTH_M),
    "bottom_right": (PITCH_LENGTH_M, PITCH_WIDTH_M),
    "center": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M / 2.0),
    "center_top": (PITCH_LENGTH_M / 2.0, 0.0),
    "center_bottom": (PITCH_LENGTH_M / 2.0, PITCH_WIDTH_M),
}

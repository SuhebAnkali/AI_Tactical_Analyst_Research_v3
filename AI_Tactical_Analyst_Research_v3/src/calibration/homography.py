"""Automatic and manual homography estimation for football pitch coordinates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.calibration.pitch import PITCH_KEYPOINTS
from src.core.config import HomographyConfig
from src.core.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from src.utils.image_ops import generate_pitch_mask


@dataclass(slots=True)
class HomographyState:
    matrix: np.ndarray | None = None
    inverse_matrix: np.ndarray | None = None
    source: str = "uninitialized"
    valid: bool = False


class HomographyEstimator:
    """Estimate homography from broadcast image to canonical pitch space.

    Automatic mode uses green-pitch masking + line extraction to estimate an
    approximate field quadrilateral. Manual mode loads correspondences from a
    JSON file and computes a precise transform with RANSAC.
    """

    def __init__(self, config: HomographyConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.state = HomographyState()
        self.manual_calibration_path = Path(config.manual_calibration_path)
        if self.manual_calibration_path.exists() and not config.auto_calibrate:
            self.load_manual_calibration(self.manual_calibration_path)

    def load_manual_calibration(self, path: str | Path) -> None:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not payload.get("enabled", True):
            raise ValueError("Manual calibration file is disabled. Set enabled=true to use it.")
        src_pts = np.array(payload["image_points"], dtype=np.float32)
        dst_pts = np.array(payload["pitch_points"], dtype=np.float32)
        if len(src_pts) < 4 or len(dst_pts) < 4:
            raise ValueError("Manual calibration requires at least four point pairs")
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.config.ransac_reproj_threshold)
        if homography is None:
            raise RuntimeError("Failed to estimate manual homography")
        self.state = HomographyState(matrix=homography, inverse_matrix=np.linalg.inv(homography), source="manual", valid=True)

    def maybe_refresh(self, frame: np.ndarray, frame_index: int) -> HomographyState:
        if self.state.valid and self.state.source == "manual":
            return self.state
        if not self.config.auto_calibrate:
            return self.state
        if self.state.valid and frame_index % self.config.refresh_interval_frames != 0:
            return self.state
        estimated = self.estimate_automatic(frame)
        if estimated.valid:
            self.state = estimated
        return self.state

    def estimate_automatic(self, frame: np.ndarray) -> HomographyState:
        try:
            pitch_mask = generate_pitch_mask(frame)
            contours, _ = cv2.findContours(pitch_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return self.state
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(polygon) < 4:
                rect = cv2.minAreaRect(contour)
                polygon = cv2.boxPoints(rect).astype(np.float32)
            else:
                polygon = polygon.reshape(-1, 2).astype(np.float32)
            hull = cv2.convexHull(polygon).reshape(-1, 2)
            if len(hull) < 4:
                return self.state
            quad = self._ordered_quad_from_hull(hull)
            dst = np.array([
                [0.0, 0.0],
                [PITCH_LENGTH_M, 0.0],
                [PITCH_LENGTH_M, PITCH_WIDTH_M],
                [0.0, PITCH_WIDTH_M],
            ], dtype=np.float32)
            homography, _ = cv2.findHomography(quad, dst, cv2.RANSAC, self.config.ransac_reproj_threshold)
            if homography is None:
                return self.state
            inv = np.linalg.inv(homography)
            return HomographyState(matrix=homography, inverse_matrix=inv, source="automatic", valid=True)
        except Exception as exc:
            self.logger.warning("Automatic homography failed: %s", exc)
            return self.state

    def save_calibration(self, path: str | Path) -> None:
        if not self.state.valid or self.state.matrix is None:
            raise RuntimeError("No valid homography state to save")
        payload = {
            "homography": self.state.matrix.tolist(),
            "source": self.state.source,
            "pitch_reference_points": PITCH_KEYPOINTS,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _ordered_quad_from_hull(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        return rect

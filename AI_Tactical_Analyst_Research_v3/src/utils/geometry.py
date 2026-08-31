"""Geometry helpers for vision and pitch-space reasoning."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from shapely.geometry import Point, Polygon


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def pairwise_iou(boxes_a: list[np.ndarray], boxes_b: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU matrix between two box lists."""
    matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            matrix[i, j] = bbox_iou(box_a, box_b)
    return matrix


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip a box to image bounds."""
    clipped = box.astype(float).copy()
    clipped[0] = np.clip(clipped[0], 0, width - 1)
    clipped[1] = np.clip(clipped[1], 0, height - 1)
    clipped[2] = np.clip(clipped[2], 0, width - 1)
    clipped[3] = np.clip(clipped[3], 0, height - 1)
    return clipped


def bottom_center(box: np.ndarray) -> tuple[float, float]:
    """Return bottom-center pixel coordinate."""
    return (float((box[0] + box[2]) / 2.0), float(box[3]))


def center(box: np.ndarray) -> tuple[float, float]:
    """Return center pixel coordinate."""
    return (float((box[0] + box[2]) / 2.0), float((box[1] + box[3]) / 2.0))


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between 2D points."""
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def moving_average(values: Iterable[float], window: int) -> list[float]:
    """Simple moving average."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return []
    out: list[float] = []
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        out.append(float(arr[start : idx + 1].mean()))
    return out


def perspective_transform(points: list[tuple[float, float]], homography: np.ndarray | None) -> list[tuple[float, float] | None]:
    """Transform image points into pitch coordinates."""
    if homography is None or not points:
        return [None for _ in points]
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in transformed]


def inverse_perspective_transform(points: list[tuple[float, float]], homography_inv: np.ndarray | None) -> list[tuple[int, int] | None]:
    """Transform pitch points back into image space."""
    if homography_inv is None or not points:
        return [None for _ in points]
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, homography_inv).reshape(-1, 2)
    return [(int(round(x)), int(round(y))) for x, y in transformed]


def inside_polygon(point_xy: tuple[float, float], polygon_xy: list[tuple[float, float]]) -> bool:
    """Check if a 2D point lies inside a polygon."""
    return Polygon(polygon_xy).contains(Point(point_xy))

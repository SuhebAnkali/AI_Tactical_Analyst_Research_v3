"""Image processing helper utilities."""

from __future__ import annotations

import cv2
import numpy as np


def extract_crop(frame: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
    """Extract a clipped crop from a frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2].copy()


def shirt_crop(frame: np.ndarray, box_xyxy: np.ndarray, top_ratio: float, bottom_ratio: float) -> np.ndarray:
    """Extract upper-body shirt crop from a player box."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    height = max(1, y2 - y1)
    crop_top = y1 + int(height * top_ratio)
    crop_bottom = y1 + int(height * bottom_ratio)
    return extract_crop(frame, np.array([x1, crop_top, x2, crop_bottom], dtype=float))


def remove_grass_pixels(bgr_crop: np.ndarray, hsv_low: tuple[int, int, int], hsv_high: tuple[int, int, int]) -> np.ndarray:
    """Remove grass-colored pixels from a crop."""
    if bgr_crop.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    mask_grass = cv2.inRange(hsv, np.array(hsv_low, dtype=np.uint8), np.array(hsv_high, dtype=np.uint8))
    mask_non_grass = mask_grass == 0
    pixels = bgr_crop[mask_non_grass]
    return pixels.reshape(-1, 3)


def robust_pixel_subset(pixels_bgr: np.ndarray, z_thresh: float = 2.5) -> np.ndarray:
    """Remove color outliers with robust z-scores."""
    if pixels_bgr.size == 0 or len(pixels_bgr) < 12:
        return pixels_bgr
    pixels = pixels_bgr.astype(float)
    median = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels - median), axis=0) + 1e-6
    robust_z = 0.6745 * np.abs(pixels - median) / mad
    keep = (robust_z < z_thresh).all(axis=1)
    filtered = pixels_bgr[keep]
    return filtered if len(filtered) >= max(8, len(pixels_bgr) // 6) else pixels_bgr


def draw_text_box(frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw a translucent text label."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, x)
    y = max(h + 4, y)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - h - 8), (x + w + 8, y + 4), color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, text, (x + 4, y - 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def generate_pitch_mask(frame: np.ndarray) -> np.ndarray:
    """Estimate green pitch mask."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([28, 20, 20], dtype=np.uint8)
    high = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

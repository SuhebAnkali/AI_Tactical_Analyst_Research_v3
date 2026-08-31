"""Generate a tiny synthetic football-like demo clip for smoke testing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUTPUT = Path("outputs/sample_demo.mp4")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
width, height, fps, frames = 1280, 720, 25, 180
writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
for i in range(frames):
    frame = np.full((height, width, 3), (36, 120, 36), dtype=np.uint8)
    cv2.rectangle(frame, (80, 80), (1200, 640), (255, 255, 255), 3)
    cv2.line(frame, (640, 80), (640, 640), (255, 255, 255), 2)
    cv2.circle(frame, (640, 360), 80, (255, 255, 255), 2)
    for idx in range(10):
        x = 220 + idx * 70 + int(18 * np.sin((i + idx) / 10))
        y = 220 + (idx % 5) * 65 + int(10 * np.cos((i + idx) / 8))
        cv2.rectangle(frame, (x, y), (x + 26, y + 54), (255, 0, 0), -1)
    for idx in range(10):
        x = 760 + idx * 35 + int(15 * np.sin((i + idx) / 9))
        y = 180 + (idx % 5) * 72 + int(12 * np.cos((i + idx) / 11))
        cv2.rectangle(frame, (x, y), (x + 26, y + 54), (0, 220, 255), -1)
    ball_x = 250 + int((i / frames) * 750)
    ball_y = 350 + int(50 * np.sin(i / 15))
    cv2.circle(frame, (ball_x, ball_y), 8, (240, 240, 240), -1)
    writer.write(frame)
writer.release()
print(f"Synthetic demo saved to {OUTPUT}")

"""Video I/O helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(slots=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class VideoReader:
    """Thin wrapper around OpenCV video capture."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.metadata = VideoMetadata(
            path=self.video_path,
            width=width,
            height=height,
            fps=float(fps),
            frame_count=frame_count,
            duration_seconds=float(frame_count / fps) if fps > 0 else 0.0,
        )

    def __iter__(self):
        frame_index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1

    def release(self) -> None:
        self.capture.release()


class VideoWriter:
    """OpenCV MP4 video writer wrapper."""

    def __init__(self, output_path: str | Path, width: int, height: int, fps: float) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to create video writer: {self.output_path}")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()

"""Annotated video frame renderer."""

from __future__ import annotations

import cv2
import numpy as np

from src.core.types import EventRecord, FormationSnapshot, TrackState
from src.utils.image_ops import draw_text_box
from src.visualization.tactical_board import TEAM_COLORS, TacticalBoardRenderer


class VideoAnnotator:
    """Draw player IDs, teams, speed, ball trail, FPS and tactical inset."""

    def __init__(self) -> None:
        self.board = TacticalBoardRenderer()
        self.ball_trail: list[tuple[int, int]] = []

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[TrackState],
        ball_track: TrackState | None,
        formations: list[FormationSnapshot],
        events: list[EventRecord],
        frame_index: int,
        fps: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox_xyxy]
            color = TEAM_COLORS.get(track.team_id, TEAM_COLORS["unknown"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.track_id} | {track.class_name} | {track.team_id}"
            if track.smoothed_speed_mps > 0:
                label += f" | {track.smoothed_speed_mps:.1f}m/s"
            draw_text_box(canvas, label, (x1, y1 - 6), color)
        if ball_track is not None:
            center = (int((ball_track.bbox_xyxy[0] + ball_track.bbox_xyxy[2]) / 2), int((ball_track.bbox_xyxy[1] + ball_track.bbox_xyxy[3]) / 2))
            self.ball_trail.append(center)
            self.ball_trail = self.ball_trail[-25:]
            for idx in range(1, len(self.ball_trail)):
                thickness = max(1, idx // 5)
                cv2.line(canvas, self.ball_trail[idx - 1], self.ball_trail[idx], (240, 240, 240), thickness)
            cv2.circle(canvas, center, 6, (255, 255, 255), -1)
            cv2.circle(canvas, center, 7, (20, 20, 20), 1)
        header = f"Frame: {frame_index} | FPS: {fps:.2f}"
        draw_text_box(canvas, header, (12, 24), (60, 60, 60))
        if events:
            last_events = ", ".join(sorted({event.event_type for event in events})[:4])
            draw_text_box(canvas, f"Events: {last_events}", (12, 52), (80, 50, 140))
        tactical = self.board.render(tracks, ball_track, formations)
        tactical = cv2.resize(tactical, (int(canvas.shape[1] * 0.34), int(canvas.shape[0] * 0.34)))
        h, w = tactical.shape[:2]
        y1 = canvas.shape[0] - h - 12
        x1 = canvas.shape[1] - w - 12
        canvas[y1 : y1 + h, x1 : x1 + w] = cv2.addWeighted(canvas[y1 : y1 + h, x1 : x1 + w], 0.15, tactical, 0.85, 0)
        return canvas

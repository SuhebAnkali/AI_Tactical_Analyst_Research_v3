"""Live tactical board rendering."""

from __future__ import annotations

import cv2
import numpy as np

from src.calibration.pitch import PitchCanvas
from src.core.types import FormationSnapshot, TrackState

TEAM_COLORS = {
    "team_home": (71, 99, 255),
    "team_away": (255, 179, 71),
    "goalkeeper_home": (180, 105, 255),
    "goalkeeper_away": (0, 255, 255),
    "referee": (220, 220, 220),
    "unknown": (160, 160, 160),
}


class TacticalBoardRenderer:
    """Render a top-down tactical board for one frame."""

    def __init__(self) -> None:
        self.pitch = PitchCanvas(width_px=760, height_px=500)

    def render(self, tracks: list[TrackState], ball_track: TrackState | None, formations: list[FormationSnapshot]) -> np.ndarray:
        board = self.pitch.blank()
        for track in tracks:
            if track.pitch_xy is None:
                continue
            center = self.pitch.to_canvas(track.pitch_xy)
            color = TEAM_COLORS.get(track.team_id, TEAM_COLORS["unknown"])
            radius = 9 if track.class_name == "player" else 11
            if track.class_name == "referee":
                radius = 7
            cv2.circle(board, center, radius, color, -1)
            cv2.circle(board, center, radius, (25, 25, 25), 2)
            cv2.putText(board, str(track.track_id), (center[0] + 8, center[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if ball_track is not None and ball_track.pitch_xy is not None:
            ball_pt = self.pitch.to_canvas(ball_track.pitch_xy)
            cv2.circle(board, ball_pt, 5, (245, 245, 245), -1)
            cv2.circle(board, ball_pt, 6, (20, 20, 20), 1)
        y = 24
        for snapshot in formations:
            text = f"{snapshot.team_id}: {snapshot.formation} ({'-'.join(map(str, snapshot.line_counts))})"
            cv2.putText(board, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
            y += 24
        return board

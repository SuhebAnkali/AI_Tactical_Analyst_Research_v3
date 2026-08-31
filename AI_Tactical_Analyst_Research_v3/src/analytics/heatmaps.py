"""Heatmap generation for players, teams, ball, and possession."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from src.core.config import HeatmapConfig
from src.core.constants import HEATMAP_DIR, PITCH_LENGTH_M, PITCH_WIDTH_M
from src.core.types import TrackState


class HeatmapEngine:
    """Collect pitch coordinates and render heatmaps."""

    def __init__(self, config: HeatmapConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.player_points: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
        self.team_points: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
        self.ball_points: list[tuple[float, float]] = []
        self.possession_points: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)

    def update(self, tracks: list[TrackState], ball_track: TrackState | None, possessing_track_id: int | None) -> None:
        owner_team = None
        for track in tracks:
            if track.pitch_xy is None:
                continue
            if track.class_name in {"player", "goalkeeper"}:
                self.player_points[track.track_id].append(track.pitch_xy)
                self.team_points[track.team_id].append(track.pitch_xy)
            if possessing_track_id is not None and track.track_id == possessing_track_id:
                owner_team = track.team_id
        if ball_track is not None and ball_track.pitch_xy is not None:
            self.ball_points.append(ball_track.pitch_xy)
            if owner_team is not None:
                self.possession_points[owner_team].append(ball_track.pitch_xy)

    def render_all(self) -> list[Path]:
        outputs: list[Path] = []
        for track_id, points in self.player_points.items():
            outputs.append(self._render(points, HEATMAP_DIR / f"player_{track_id}_heatmap.png", title=f"Player {track_id} Heatmap"))
        for team_id, points in self.team_points.items():
            outputs.append(self._render(points, HEATMAP_DIR / f"{team_id}_heatmap.png", title=f"{team_id} Heatmap"))
        outputs.append(self._render(self.ball_points, HEATMAP_DIR / "ball_heatmap.png", title="Ball Heatmap"))
        for team_id, points in self.possession_points.items():
            outputs.append(self._render(points, HEATMAP_DIR / f"{team_id}_possession_heatmap.png", title=f"{team_id} Possession Heatmap"))
        return [path for path in outputs if path.exists()]

    def _render(self, points: list[tuple[float, float]], path: Path, title: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8), dpi=160)
        heatmap = np.zeros((self.config.bins_y, self.config.bins_x), dtype=float)
        if points:
            xs = np.clip([p[0] for p in points], 0, PITCH_LENGTH_M)
            ys = np.clip([p[1] for p in points], 0, PITCH_WIDTH_M)
            heatmap, _, _ = np.histogram2d(ys, xs, bins=[self.config.bins_y, self.config.bins_x], range=[[0, PITCH_WIDTH_M], [0, PITCH_LENGTH_M]])
            heatmap = gaussian_filter(heatmap, sigma=self.config.gaussian_sigma)
        ax.imshow(heatmap, extent=[0, PITCH_LENGTH_M, PITCH_WIDTH_M, 0], cmap="inferno", alpha=0.92, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Pitch length (m)")
        ax.set_ylabel("Pitch width (m)")
        ax.set_facecolor("#1a472a")
        ax.set_xlim(0, PITCH_LENGTH_M)
        ax.set_ylim(PITCH_WIDTH_M, 0)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    def inventory_frame(self) -> pd.DataFrame:
        rows = []
        for track_id, points in self.player_points.items():
            rows.append({"entity_type": "player", "entity_id": track_id, "samples": len(points)})
        for team_id, points in self.team_points.items():
            rows.append({"entity_type": "team", "entity_id": team_id, "samples": len(points)})
        rows.append({"entity_type": "ball", "entity_id": "ball", "samples": len(self.ball_points)})
        return pd.DataFrame(rows)

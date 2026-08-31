"""Formation recognition from pitch-space player positions."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.core.config import FormationConfig
from src.core.constants import FORMATION_TEMPLATES, PITCH_LENGTH_M
from src.core.types import FormationSnapshot, TrackState


@dataclass
class _FormationHistory:
    labels: deque[str] = field(default_factory=deque)
    line_histories: deque[list[int]] = field(default_factory=deque)

    def update(self, formation: str, line_counts: list[int], window: int) -> tuple[str, list[int]]:
        self.labels.append(formation)
        self.line_histories.append(line_counts)
        while len(self.labels) > window:
            self.labels.popleft()
        while len(self.line_histories) > window:
            self.line_histories.popleft()
        winner = Counter(self.labels).most_common(1)[0][0]
        candidates = [line for line, label in zip(self.line_histories, self.labels, strict=False) if label == winner]
        smoothed_lines = max(candidates, key=lambda row: len(row)) if candidates else line_counts
        return winner, smoothed_lines


class FormationRecognizer:
    """Infer common formations with clustering and smoothing."""

    def __init__(self, config: FormationConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.histories: dict[str, _FormationHistory] = defaultdict(_FormationHistory)

    def update(self, frame_index: int, tracks: list[TrackState]) -> list[FormationSnapshot]:
        snapshots: list[FormationSnapshot] = []
        for team_id in ["team_home", "team_away"]:
            players = [track for track in tracks if track.team_id == team_id and track.class_name == "player" and track.pitch_xy is not None]
            if len(players) < self.config.min_players:
                continue
            coords = np.array([track.pitch_xy for track in players], dtype=float)
            direction = self._infer_direction(team_id, coords)
            line_counts, confidence = self._estimate_lines(coords, direction)
            formation = self._closest_template(line_counts)
            smooth_form, smooth_lines = self.histories[team_id].update(formation, line_counts, self.config.smoothing_window)
            snapshots.append(
                FormationSnapshot(
                    frame_index=frame_index,
                    team_id=team_id,
                    formation=smooth_form,
                    line_counts=smooth_lines,
                    confidence=confidence,
                    direction=direction,
                )
            )
        return snapshots

    def _infer_direction(self, team_id: str, coords: np.ndarray) -> str:
        mean_x = float(coords[:, 0].mean())
        if team_id == "team_home":
            return "left_to_right" if mean_x < PITCH_LENGTH_M / 2.0 else "right_to_left"
        return "right_to_left" if mean_x < PITCH_LENGTH_M / 2.0 else "left_to_right"

    def _estimate_lines(self, coords: np.ndarray, direction: str) -> tuple[list[int], float]:
        xs = coords[:, 0] if direction == "left_to_right" else (PITCH_LENGTH_M - coords[:, 0])
        n_players = len(xs)
        candidate_counts = [3, 4]
        best_lines = [4, 4, 2]
        best_score = float("inf")

        for n_lines in candidate_counts:
            clustering = AgglomerativeClustering(n_clusters=n_lines)
            labels = clustering.fit_predict(xs.reshape(-1, 1))
            means = [(label, float(xs[labels == label].mean())) for label in sorted(set(labels))]
            means.sort(key=lambda item: item[1])
            ordered_labels = [label for label, _ in means]
            line_counts = [int(np.sum(labels == label)) for label in ordered_labels]
            if sum(line_counts) != n_players:
                continue
            score = self._formation_compactness(xs, labels)
            if score < best_score:
                best_score = score
                best_lines = line_counts
        confidence = 1.0 / (1.0 + best_score)
        return best_lines, float(np.clip(confidence, 0.0, 1.0))

    def _formation_compactness(self, xs: np.ndarray, labels: np.ndarray) -> float:
        score = 0.0
        for label in set(labels):
            subset = xs[labels == label]
            if len(subset) <= 1:
                continue
            score += float(np.var(subset))
        return score / max(1, len(set(labels)))

    def _closest_template(self, line_counts: list[int]) -> str:
        best_name = "4-4-2"
        best_distance = float("inf")
        candidate = np.array(line_counts, dtype=float)
        for name, template in FORMATION_TEMPLATES.items():
            tpl = np.array(template, dtype=float)
            if len(tpl) != len(candidate):
                penalty = abs(len(tpl) - len(candidate)) * 2.0
                overlap = min(len(tpl), len(candidate))
                distance = np.linalg.norm(tpl[:overlap] - candidate[:overlap]) + penalty
            else:
                distance = np.linalg.norm(tpl - candidate)
            if distance < best_distance:
                best_distance = float(distance)
                best_name = name
        return best_name

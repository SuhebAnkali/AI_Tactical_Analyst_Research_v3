"""Research-oriented team classification using shirt-color embeddings."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

from src.core.config import TeamClassifierConfig
from src.core.types import TrackState
from src.utils.image_ops import remove_grass_pixels, robust_pixel_subset, shirt_crop


@dataclass
class _TrackLabelHistory:
    labels: deque[str] = field(default_factory=lambda: deque(maxlen=30))

    def update(self, label: str) -> str:
        self.labels.append(label)
        return Counter(self.labels).most_common(1)[0][0]


class TeamClassifier:
    """Advanced jersey-color based team classification.

    The pipeline combines shirt crop extraction, grass masking, LAB/HSV features,
    KMeans color decomposition, outlier removal, temporal smoothing, and global
    prototype stabilization so players do not oscillate between team assignments.
    Goalkeepers are promoted to dedicated labels, while referees stay isolated.
    """

    def __init__(self, config: TeamClassifierConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.track_histories: dict[int, _TrackLabelHistory] = defaultdict(_TrackLabelHistory)
        self.track_feature_history: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=config.temporal_window))
        self.team_prototypes: dict[str, np.ndarray] = {}
        self.initialized = False

    def classify(self, frame: np.ndarray, tracks: list[TrackState]) -> dict[int, str]:
        """Assign stable team labels to active tracks."""
        feature_rows: list[np.ndarray] = []
        track_ids: list[int] = []
        prelabels: dict[int, str] = {}

        for track in tracks:
            if track.class_name == "referee":
                prelabels[track.track_id] = "referee"
                continue
            feature = self._extract_uniform_feature(frame, track)
            if feature is None:
                prelabels[track.track_id] = track.team_id if track.team_id != "unknown" else "unknown"
                continue
            self.track_feature_history[track.track_id].append(feature)
            smoothed_feature = np.mean(np.stack(list(self.track_feature_history[track.track_id])), axis=0)
            if track.class_name == "goalkeeper":
                prelabels[track.track_id] = "unknown"
                track.metadata["uniform_feature"] = smoothed_feature
                continue
            feature_rows.append(smoothed_feature)
            track_ids.append(track.track_id)
            track.metadata["uniform_feature"] = smoothed_feature

        if len(feature_rows) >= 4:
            self._update_team_prototypes(np.stack(feature_rows))

        if self.team_prototypes:
            labels = self._assign_prototypes(track_ids, feature_rows)
            prelabels.update(labels)

        goalkeepers = [track for track in tracks if track.class_name == "goalkeeper"]
        self._assign_goalkeepers(goalkeepers, prelabels)

        stabilized: dict[int, str] = {}

        for track in tracks:
            candidate = prelabels.get(
                track.track_id,
                track.team_id if track.team_id != "unknown" else "unknown",
            )

            if track.class_name == "referee":
                track.team_id = "referee"
                stabilized[track.track_id] = "referee"
                continue

            if track.team_id in {
                "team_home",
                "team_away",
                "goalkeeper_home",
                "goalkeeper_away",
            }:
                stabilized[track.track_id] = track.team_id
                continue

            if candidate == "unknown":
                stabilized[track.track_id] = "unknown"
                continue

            stable = self.track_histories[track.track_id].update(candidate)

            if stable != "unknown":
                track.team_id = stable

            stabilized[track.track_id] = track.team_id

        return stabilized
    def _extract_uniform_feature(self, frame: np.ndarray, track: TrackState) -> np.ndarray | None:
        crop = shirt_crop(frame, track.bbox_xyxy, self.config.shirt_crop_top_ratio, self.config.shirt_crop_bottom_ratio)
        pixels_bgr = remove_grass_pixels(crop, self.config.grass_hsv_low, self.config.grass_hsv_high)
        pixels_bgr = robust_pixel_subset(pixels_bgr)
        if pixels_bgr.size == 0 or len(pixels_bgr) < self.config.min_pixels:
            return None

        pixels_bgr = pixels_bgr.astype(np.uint8)
        bgr_for_kmeans = pixels_bgr.reshape(-1, 3).astype(np.float32)
        k = min(self.config.kmeans_clusters, len(bgr_for_kmeans))
        kmeans = KMeans(n_clusters=max(1, k), random_state=42, n_init=5)
        cluster_ids = kmeans.fit_predict(bgr_for_kmeans)
        counts = Counter(cluster_ids)

        cluster_features: list[np.ndarray] = []
        cluster_weights: list[float] = []
        for cluster_idx, _ in counts.most_common():
            subset = pixels_bgr[cluster_ids == cluster_idx]
            if len(subset) < 10:
                continue
            hsv = cv2.cvtColor(subset.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            lab = cv2.cvtColor(subset.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
            feature = np.concatenate([
                subset.mean(axis=0),
                subset.std(axis=0),
                hsv.mean(axis=0),
                lab.mean(axis=0),
            ]).astype(float)
            cluster_features.append(feature)
            cluster_weights.append(float(len(subset)))
        if not cluster_features:
            return None
        stacked = np.stack(cluster_features)
        weights = np.array(cluster_weights, dtype=float)
        weights = weights / weights.sum()
        return np.average(stacked, axis=0, weights=weights)

    def _update_team_prototypes(self, features: np.ndarray) -> None:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init= "auto")
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_
        if not self.initialized:
            self.team_prototypes["team_home"] = centers[0]
            self.team_prototypes["team_away"] = centers[1]
            self.initialized = True
            return

        old_centers = np.stack([self.team_prototypes["team_home"], self.team_prototypes["team_away"]])
        nearest = pairwise_distances_argmin(centers, old_centers)
        for src_idx, old_idx in enumerate(nearest):
            label = "team_home" if old_idx == 0 else "team_away"
            self.team_prototypes[label] = (
                self.config.color_history_decay * self.team_prototypes[label]
                + (1.0 - self.config.color_history_decay) * centers[src_idx]
            )

        if len(set(labels)) == 2:
            for label_name, label_idx in [("team_home", 0), ("team_away", 1)]:
                subset = features[labels == label_idx]
                if len(subset) == 0:
                    continue
                centroid = subset.mean(axis=0)
                self.team_prototypes[label_name] = 0.9 * self.team_prototypes[label_name] + 0.1 * centroid

    def _assign_prototypes(self, track_ids: list[int], features: list[np.ndarray]) -> dict[int, str]:
        if not features or len(self.team_prototypes) < 2:
            return {}
        prototypes = np.stack([self.team_prototypes["team_home"], self.team_prototypes["team_away"]])
        feature_array = np.stack(features)
        indices = pairwise_distances_argmin(feature_array, prototypes)
        labels = ["team_home" if idx == 0 else "team_away" for idx in indices]
        return dict(zip(track_ids, labels, strict=False))

    def _assign_goalkeepers(self, goalkeepers: list[TrackState], labels: dict[int, str]) -> None:
        if not goalkeepers:
            return
        if len(self.team_prototypes) >= 2:
            for goalkeeper in goalkeepers:
                feature = goalkeeper.metadata.get("uniform_feature")
                if feature is None:
                    labels[goalkeeper.track_id] = goalkeeper.team_id if goalkeeper.team_id != "unknown" else "goalkeeper_home"
                    continue
                prototypes = np.stack([self.team_prototypes["team_home"], self.team_prototypes["team_away"]])
                idx = int(pairwise_distances_argmin(np.asarray(feature).reshape(1, -1), prototypes)[0])
                labels[goalkeeper.track_id] = "goalkeeper_home" if idx == 0 else "goalkeeper_away"
        else:
            for idx, goalkeeper in enumerate(sorted(goalkeepers, key=lambda t: t.bbox_xyxy[0])):
                labels[goalkeeper.track_id] = "goalkeeper_home" if idx == 0 else "goalkeeper_away"

"""ByteTrack-inspired multi-object tracker with improved ID persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.config import TrackerConfig
from src.core.types import Detection, TrackState
from src.utils.geometry import bottom_center, pairwise_iou


@dataclass
class _TrackBucket:
    """Internal storage for active, lost and removed tracks."""

    active: dict[int, TrackState] = field(default_factory=dict)
    lost: dict[int, TrackState] = field(default_factory=dict)
    removed: dict[int, TrackState] = field(default_factory=dict)


class ByteTracker:
    """ByteTrack-style tracker tuned for football videos.

    Features:
    - High- and low-confidence matching
    - Velocity-based position prediction
    - Lost-track reactivation
    - Class-aware association
    - Separate ball tracking
    - Persistent TrackState objects and team labels
    """

    def __init__(
        self,
        config: TrackerConfig,
        fps: float,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.fps = max(float(fps), 1.0)
        self.logger = logger

        self.next_track_id = 1

        self.entities = _TrackBucket()
        self.ball_tracks = _TrackBucket()

    def update(
        self,
        detections: list[Detection],
        frame_index: int,
    ) -> tuple[list[TrackState], TrackState | None]:
        """Update player-like and ball tracks for one frame."""

        entity_detections = [
            detection
            for detection in detections
            if detection.class_name != "ball"
        ]

        ball_detections = [
            detection
            for detection in detections
            if detection.class_name == "ball"
        ]

        self._update_bucket(
            self.entities,
            entity_detections,
            frame_index,
            is_ball=False,
        )

        self._update_bucket(
            self.ball_tracks,
            ball_detections,
            frame_index,
            is_ball=True,
        )

        active_tracks = sorted(
            self.entities.active.values(),
            key=lambda track: track.track_id,
        )

        ball_track: TrackState | None = None

        if self.ball_tracks.active:
            ball_track = max(
                self.ball_tracks.active.values(),
                key=lambda track: (
                    track.hits,
                    track.confidence,
                    -track.missed,
                ),
            )

        return active_tracks, ball_track

    def _update_bucket(
        self,
        bucket: _TrackBucket,
        detections: list[Detection],
        frame_index: int,
        is_ball: bool,
    ) -> None:
        """Update one tracking bucket using staged association."""

        if is_ball:
            high_detections = [
                detection
                for detection in detections
                if detection.confidence >= self.config.low_thresh
            ]

            low_detections: list[Detection] = []

        else:
            high_detections = [
                detection
                for detection in detections
                if detection.confidence >= self.config.high_thresh
            ]

            low_detections = [
                detection
                for detection in detections
                if (
                    self.config.low_thresh
                    <= detection.confidence
                    < self.config.high_thresh
                )
            ]

        active_tracks = list(bucket.active.values())
        lost_tracks = list(bucket.lost.values())

        # Stage 1: match active tracks with high-confidence detections.
        active_matches = self._match_tracks(
            active_tracks,
            high_detections,
            is_ball=is_ball,
            recovery=False,
        )

        matched_active_ids = set(active_matches.keys())
        matched_high_indices = set(active_matches.values())

        unmatched_active_tracks = [
            track
            for track in active_tracks
            if track.track_id not in matched_active_ids
        ]

        remaining_high_detections = [
            detection
            for index, detection in enumerate(high_detections)
            if index not in matched_high_indices
        ]

        # Stage 2: recover lost tracks before creating new IDs.
        lost_matches = self._match_tracks(
            lost_tracks,
            remaining_high_detections,
            is_ball=is_ball,
            recovery=True,
        )

        matched_lost_indices = set(lost_matches.values())

        remaining_high_after_recovery = [
            detection
            for index, detection in enumerate(
                remaining_high_detections
            )
            if index not in matched_lost_indices
        ]

        # Stage 3: recover unmatched active tracks using low-confidence boxes.
        low_matches = self._match_tracks(
            unmatched_active_tracks,
            low_detections,
            is_ball=is_ball,
            recovery=True,
        )

        new_active: dict[int, TrackState] = {}
        new_lost: dict[int, TrackState] = {}

        # Update active tracks.
        for track in active_tracks:

            if track.track_id in active_matches:
                detection_index = active_matches[track.track_id]
                detection = high_detections[detection_index]

                updated = self._update_track(
                    track,
                    detection,
                    frame_index,
                )

                new_active[updated.track_id] = updated
                continue

            if track.track_id in low_matches:
                detection_index = low_matches[track.track_id]
                detection = low_detections[detection_index]

                updated = self._update_track(
                    track,
                    detection,
                    frame_index,
                )

                new_active[updated.track_id] = updated
                continue

            self._mark_missed(track)

            if track.missed <= self.config.max_time_lost:
                new_lost[track.track_id] = track
            else:
                bucket.removed[track.track_id] = track

        # Reactivate lost tracks.
        for track in lost_tracks:

            if track.track_id in lost_matches:
                detection_index = lost_matches[track.track_id]
                detection = remaining_high_detections[
                    detection_index
                ]

                updated = self._update_track(
                    track,
                    detection,
                    frame_index,
                )

                new_active[updated.track_id] = updated
                continue

            self._mark_missed(track)

            if track.missed <= self.config.max_time_lost:
                new_lost[track.track_id] = track
            else:
                bucket.removed[track.track_id] = track

        # Create new IDs only after all recovery attempts.
        for detection in remaining_high_after_recovery:

            if detection.confidence < self.config.new_track_thresh:
                continue

            new_track = self._create_track(
                detection,
                frame_index,
            )

            new_active[new_track.track_id] = new_track

        bucket.active = new_active
        bucket.lost = new_lost

    def _match_tracks(
        self,
        tracks: list[TrackState],
        detections: list[Detection],
        is_ball: bool,
        recovery: bool,
    ) -> dict[int, int]:
        """Match tracks and detections using class-aware IoU."""

        if not tracks or not detections:
            return {}

        predicted_boxes = self._predict_boxes(
            tracks,
            recovery=recovery,
        )

        detection_boxes = [
            detection.bbox_xyxy
            for detection in detections
        ]

        iou_matrix = pairwise_iou(
            predicted_boxes,
            detection_boxes,
        )

        if iou_matrix.size == 0:
            return {}

        cost_matrix = 1.0 - iou_matrix

        # Prevent player/referee/goalkeeper cross-matching.
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):

                if track.class_name != detection.class_name:
                    cost_matrix[row, column] = 1_000_000.0

        row_indices, column_indices = linear_sum_assignment(
            cost_matrix
        )

        base_threshold = (
            self.config.ball_match_iou_thresh
            if is_ball
            else self.config.match_iou_thresh
        )

        minimum_iou = (
            max(0.05, base_threshold * 0.65)
            if recovery
            else base_threshold
        )

        matches: dict[int, int] = {}

        for row, column in zip(
            row_indices,
            column_indices,
            strict=False,
        ):

            if cost_matrix[row, column] >= 100_000:
                continue

            if iou_matrix[row, column] < minimum_iou:
                continue

            if not self._passes_distance_gate(
                tracks[row],
                detections[column],
                recovery=recovery,
                is_ball=is_ball,
            ):
                continue

            matches[tracks[row].track_id] = int(column)

        return matches

    def _predict_boxes(
        self,
        tracks: list[TrackState],
        recovery: bool,
    ) -> list[np.ndarray]:
        """Predict box positions using smoothed velocity."""

        predicted_boxes: list[np.ndarray] = []

        for track in tracks:

            prediction_frames = (
                max(1, track.missed + 1)
                if recovery
                else 1
            )

            velocity = np.asarray(
                track.velocity_xy,
                dtype=float,
            )

            offset = velocity * float(prediction_frames)

            predicted_box = (
                track.bbox_xyxy
                + np.array(
                    [
                        offset[0],
                        offset[1],
                        offset[0],
                        offset[1],
                    ],
                    dtype=float,
                )
            )

            predicted_boxes.append(predicted_box)

        return predicted_boxes

    def _passes_distance_gate(
        self,
        track: TrackState,
        detection: Detection,
        recovery: bool,
        is_ball: bool,
    ) -> bool:
        """Reject associations with unrealistic displacement."""

        track_center = np.asarray(
            bottom_center(track.bbox_xyxy),
            dtype=float,
        )

        detection_center = np.asarray(
            bottom_center(detection.bbox_xyxy),
            dtype=float,
        )

        displacement = float(
            np.linalg.norm(
                detection_center - track_center
            )
        )

        width = max(
            1.0,
            float(
                track.bbox_xyxy[2]
                - track.bbox_xyxy[0]
            ),
        )

        height = max(
            1.0,
            float(
                track.bbox_xyxy[3]
                - track.bbox_xyxy[1]
            ),
        )

        box_scale = max(width, height)

        multiplier = 5.0 if is_ball else 2.5

        if recovery:
            multiplier *= max(
                1.5,
                track.missed + 1,
            )

        return displacement <= multiplier * box_scale

    def _create_track(
        self,
        detection: Detection,
        frame_index: int,
    ) -> TrackState:
        """Create a new track from an unmatched detection."""

        track = TrackState(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            bbox_xyxy=detection.bbox_xyxy.copy(),
            confidence=detection.confidence,
            frame_index=frame_index,
        )

        track.append_position(
            bottom_center(track.bbox_xyxy),
            detection.pitch_xy,
        )

        self.next_track_id += 1

        return track

    def _update_track(
        self,
        track: TrackState,
        detection: Detection,
        frame_index: int,
    ) -> TrackState:
        """Update a track while preserving team and metadata."""

        previous_center = np.asarray(
            bottom_center(track.bbox_xyxy),
            dtype=float,
        )

        new_center = np.asarray(
            bottom_center(detection.bbox_xyxy),
            dtype=float,
        )

        measured_velocity = (
            new_center - previous_center
        )

        track.velocity_xy = (
            0.75 * track.velocity_xy
            + 0.25 * measured_velocity
        )

        track.bbox_xyxy = detection.bbox_xyxy.copy()
        track.class_id = detection.class_id
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.frame_index = frame_index

        track.age += 1
        track.hits += 1
        track.missed = 0

        track.append_position(
            tuple(new_center.tolist()),
            detection.pitch_xy,
        )

        return track

    @staticmethod
    def _mark_missed(
        track: TrackState,
    ) -> None:
        """Mark a track as unmatched for one frame."""

        track.missed += 1
        track.age += 1
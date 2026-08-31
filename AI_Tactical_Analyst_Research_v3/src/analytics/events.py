"""Heuristic event detection for tactical football analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.core.config import EventConfig
from src.core.constants import PITCH_LENGTH_M, PITCH_WIDTH_M
from src.core.types import EventRecord, TrackState
from src.utils.geometry import euclidean


@dataclass
class PossessionState:
    frame_index: int = -1
    owner_track_id: int | None = None
    owner_team_id: str = "unknown"
    ball_pitch_xy: tuple[float, float] | None = None


class EventDetector:
    """Generate ball-centric tactical events from tracks and motion."""

    def __init__(self, config: EventConfig, fps: float, logger: logging.Logger) -> None:
        self.config = config
        self.fps = fps
        self.logger = logger
        self.state = PossessionState()

    def update(self, frame_index: int, tracks: list[TrackState], ball_track: TrackState | None) -> tuple[list[EventRecord], int | None]:
        events: list[EventRecord] = []
        owner = self._infer_owner(tracks, ball_track)
        if owner is not None:
            events.append(EventRecord(frame_index, "touch", owner.team_id, owner.track_id, None, ball_track.track_id if ball_track else None, confidence=0.8))

        if self.state.owner_track_id is not None and owner is not None and owner.track_id != self.state.owner_track_id:
            if owner.team_id == self.state.owner_team_id:
                event_type = "pass"
            else:
                event_type = "interception"
                events.append(EventRecord(frame_index, "ball_recovery", owner.team_id, owner.track_id, self.state.owner_track_id, ball_track.track_id if ball_track else None, confidence=0.75))
            events.append(
                EventRecord(
                    frame_index=frame_index,
                    event_type=event_type,
                    team_id=owner.team_id,
                    primary_track_id=self.state.owner_track_id,
                    secondary_track_id=owner.track_id,
                    ball_track_id=ball_track.track_id if ball_track else None,
                    start_pitch_xy=self.state.ball_pitch_xy,
                    end_pitch_xy=ball_track.pitch_xy if ball_track and ball_track.pitch_xy else None,
                    confidence=0.75,
                )
            )

        if ball_track is not None and ball_track.pitch_xy is not None and len(ball_track.history_pitch_xy) >= 3:
            speed = self._ball_speed(ball_track)
            x, y = ball_track.pitch_xy
            towards_left_goal = x <= 10.5
            towards_right_goal = x >= PITCH_LENGTH_M - 10.5
            if speed >= self.config.shot_min_speed_mps and (x >= self.config.shot_final_third_x_m or x <= PITCH_LENGTH_M - self.config.shot_final_third_x_m):
                attacking_team = owner.team_id if owner is not None else self.state.owner_team_id
                events.append(EventRecord(frame_index, "shot", attacking_team, owner.track_id if owner else None, None, ball_track.track_id, confidence=0.7))
                if (towards_left_goal or towards_right_goal) and (PITCH_WIDTH_M / 2.0 - 12.0 <= y <= PITCH_WIDTH_M / 2.0 + 12.0):
                    events.append(EventRecord(frame_index, "goal_attempt", attacking_team, owner.track_id if owner else None, None, ball_track.track_id, confidence=0.65))

            if y <= 1.0 or y >= PITCH_WIDTH_M - 1.0:
                events.append(EventRecord(frame_index, "throw", owner.team_id if owner else "unknown", owner.track_id if owner else None, None, ball_track.track_id, confidence=0.6))
            if x <= 1.0 or x >= PITCH_LENGTH_M - 1.0:
                lane_center = abs(y - PITCH_WIDTH_M / 2.0)
                event_name = "corner" if lane_center > 15.0 else "goal_kick"
                events.append(EventRecord(frame_index, event_name, owner.team_id if owner else "unknown", owner.track_id if owner else None, None, ball_track.track_id, confidence=0.6))

        if owner is not None and self._offside_heuristic(owner, tracks):
            events.append(EventRecord(frame_index, "offside", owner.team_id, owner.track_id, None, ball_track.track_id if ball_track else None, confidence=0.55))

        self.state = PossessionState(
            frame_index=frame_index,
            owner_track_id=owner.track_id if owner else None,
            owner_team_id=owner.team_id if owner else "unknown",
            ball_pitch_xy=ball_track.pitch_xy if ball_track else None,
        )
        return events, owner.track_id if owner else None

    def _infer_owner(self, tracks: list[TrackState], ball_track: TrackState | None) -> TrackState | None:
        if ball_track is None:
            return None
        ball_point = ball_track.pitch_xy if ball_track.pitch_xy is not None else None
        if ball_point is None:
            return None
        candidates: list[tuple[float, TrackState]] = []
        for track in tracks:
            if track.class_name not in {"player", "goalkeeper"} or track.pitch_xy is None:
                continue
            distance = euclidean(track.pitch_xy, ball_point)
            if distance <= self.config.possession_radius_m:
                candidates.append((distance, track))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _ball_speed(self, ball_track: TrackState) -> float:
        if len(ball_track.history_pitch_xy) < 2:
            return 0.0
        a = ball_track.history_pitch_xy[-2]
        b = ball_track.history_pitch_xy[-1]
        return euclidean(a, b) * self.fps

    def _offside_heuristic(self, owner: TrackState, tracks: list[TrackState]) -> bool:
        if owner.pitch_xy is None or owner.team_id not in {"team_home", "team_away", "goalkeeper_home", "goalkeeper_away"}:
            return False
        same_team = [t for t in tracks if t.team_id.replace("goalkeeper_", "team_") == owner.team_id.replace("goalkeeper_", "team_") and t.pitch_xy is not None]
        opponents = [t for t in tracks if t.team_id.replace("goalkeeper_", "team_") != owner.team_id.replace("goalkeeper_", "team_") and t.team_id != "referee" and t.pitch_xy is not None]
        if len(opponents) < 2 or len(same_team) < 2:
            return False
        if owner.team_id.startswith("team_home") or owner.team_id.startswith("goalkeeper_home"):
            second_last_defender_x = sorted([t.pitch_xy[0] for t in opponents])[1]
            return owner.pitch_xy[0] > second_last_defender_x + 1.0 and owner.pitch_xy[0] > PITCH_LENGTH_M / 2.0
        second_last_defender_x = sorted([t.pitch_xy[0] for t in opponents], reverse=True)[1]
        return owner.pitch_xy[0] < second_last_defender_x - 1.0 and owner.pitch_xy[0] < PITCH_LENGTH_M / 2.0

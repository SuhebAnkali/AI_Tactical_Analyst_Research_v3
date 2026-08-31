"""Player and team statistical accumulators."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.constants import PITCH_LENGTH_M
from src.core.types import EventRecord, TrackState
from src.utils.geometry import euclidean


@dataclass
class PlayerAggregate:
    track_id: int
    class_name: str
    team_id: str
    frames_observed: int = 0
    distance_m: float = 0.0
    max_speed_mps: float = 0.0
    mean_speed_mps: float = 0.0
    sprint_count: int = 0
    touches: int = 0
    possession_frames: int = 0
    own_half_frames: int = 0
    opponent_half_frames: int = 0
    ball_recoveries: int = 0
    pass_estimation: int = 0
    last_pitch_xy: tuple[float, float] | None = None
    speed_samples: list[float] = field(default_factory=list)


class StatisticsEngine:
    """Accumulate player/team statistics frame by frame."""

    def __init__(self, fps: float, logger: logging.Logger) -> None:
        self.fps = fps
        self.logger = logger
        self.players: dict[int, PlayerAggregate] = {}
        self.team_possession_frames: defaultdict[str, int] = defaultdict(int)

    def update_tracks(self, tracks: list[TrackState], possessing_track_id: int | None) -> None:
        for track in tracks:
            if track.class_name not in {"player", "goalkeeper", "referee"}:
                continue
            aggregate = self.players.setdefault(
                track.track_id,
                PlayerAggregate(track_id=track.track_id, class_name=track.class_name, team_id=track.team_id),
            )
            aggregate.team_id = track.team_id
            aggregate.frames_observed += 1
            if track.pitch_xy is not None and aggregate.last_pitch_xy is not None:
                d_m = euclidean(track.pitch_xy, aggregate.last_pitch_xy)
                aggregate.distance_m += d_m
                speed = d_m * self.fps
                aggregate.speed_samples.append(speed)
                aggregate.max_speed_mps = max(aggregate.max_speed_mps, speed)
                if speed >= 7.0 and (len(aggregate.speed_samples) < 2 or aggregate.speed_samples[-2] < 7.0):
                    aggregate.sprint_count += 1
            if track.pitch_xy is not None:
                aggregate.last_pitch_xy = track.pitch_xy
                if track.team_id in {"team_home", "goalkeeper_home"}:
                    own_half = track.pitch_xy[0] < PITCH_LENGTH_M / 2.0
                elif track.team_id in {"team_away", "goalkeeper_away"}:
                    own_half = track.pitch_xy[0] > PITCH_LENGTH_M / 2.0
                else:
                    own_half = True
                if own_half:
                    aggregate.own_half_frames += 1
                else:
                    aggregate.opponent_half_frames += 1
            if possessing_track_id is not None and track.track_id == possessing_track_id:
                aggregate.possession_frames += 1
                self.team_possession_frames[track.team_id] += 1

            if aggregate.speed_samples:
                aggregate.mean_speed_mps = float(np.mean(aggregate.speed_samples))

    def register_events(self, events: list[EventRecord]) -> None:
        for event in events:
            if event.primary_track_id is None:
                continue
            aggregate = self.players.get(event.primary_track_id)
            if aggregate is None:
                continue
            if event.event_type == "touch":
                aggregate.touches += 1
            elif event.event_type == "ball_recovery":
                aggregate.ball_recoveries += 1
            elif event.event_type == "pass":
                aggregate.pass_estimation += 1

    def to_frame(self) -> pd.DataFrame:
        records = []
        for aggregate in self.players.values():
            records.append(
                {
                    "track_id": aggregate.track_id,
                    "class_name": aggregate.class_name,
                    "team_id": aggregate.team_id,
                    "frames_observed": aggregate.frames_observed,
                    "distance_m": round(aggregate.distance_m, 3),
                    "max_speed_mps": round(aggregate.max_speed_mps, 3),
                    "mean_speed_mps": round(aggregate.mean_speed_mps, 3),
                    "sprint_count": aggregate.sprint_count,
                    "touches": aggregate.touches,
                    "possession_seconds": round(aggregate.possession_frames / self.fps, 3),
                    "time_in_own_half_seconds": round(aggregate.own_half_frames / self.fps, 3),
                    "time_in_opponent_half_seconds": round(aggregate.opponent_half_frames / self.fps, 3),
                    "ball_recoveries": aggregate.ball_recoveries,
                    "pass_estimation": aggregate.pass_estimation,
                }
            )
        return pd.DataFrame(records).sort_values(by=["team_id", "track_id"], ignore_index=True) if records else pd.DataFrame()

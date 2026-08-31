"""Estimate football possession using image-space ownership and pitch-space zones.

Version 2 improves ownership estimation by using:
- ball bounding-box centre in image pixels,
- player foot point in image pixels,
- player-height-normalized distance,
- temporal smoothing before possession switches,
- pitch coordinates only for field-zone analysis.

Outputs:
- possession_timeline_v2.csv
- possession_summary_v2.csv
- possession_report_v2.json

Important:
Possession is an estimate derived from detections and tracking. It should be
reported as estimated possession, not official match statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PLAYER_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "track_id",
    "class_name",
    "team_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
}

BALL_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pitch_x_m",
    "pitch_y_m",
    "confidence",
}


class PossessionAnalyticsV2:
    """Estimate possession with image-space player-ball association."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        normalized_control_threshold: float = 1.5,
        switch_confirmation_frames: int = 2,
        hold_last_team_frames: int = 4,
        pitch_length_m: float = 105.0,
        minimum_player_height_px: float = 8.0,
    ) -> None:
        if normalized_control_threshold <= 0:
            raise ValueError(
                "normalized_control_threshold must be greater than zero."
            )
        if switch_confirmation_frames < 1:
            raise ValueError(
                "switch_confirmation_frames must be at least 1."
            )
        if hold_last_team_frames < 0:
            raise ValueError(
                "hold_last_team_frames cannot be negative."
            )
        if minimum_player_height_px <= 0:
            raise ValueError(
                "minimum_player_height_px must be greater than zero."
            )

        self.normalized_control_threshold = float(
            normalized_control_threshold
        )
        self.switch_confirmation_frames = int(
            switch_confirmation_frames
        )
        self.hold_last_team_frames = int(
            hold_last_team_frames
        )
        self.pitch_length_m = float(pitch_length_m)
        self.minimum_player_height_px = float(
            minimum_player_height_px
        )

    def run(
        self,
        players_csv: str | Path,
        ball_csv: str | Path,
        timeline_output_csv: str | Path,
        summary_output_csv: str | Path,
        report_output_json: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        players_path = Path(players_csv)
        ball_path = Path(ball_csv)

        if not players_path.exists():
            raise FileNotFoundError(
                f"Player tracking file not found: {players_path}"
            )
        if not ball_path.exists():
            raise FileNotFoundError(
                f"Ball tracking file not found: {ball_path}"
            )

        players = pd.read_csv(players_path)
        ball = pd.read_csv(ball_path)

        self._validate_columns(
            players,
            PLAYER_REQUIRED_COLUMNS,
            "player tracking",
        )
        self._validate_columns(
            ball,
            BALL_REQUIRED_COLUMNS,
            "ball tracking",
        )

        players = self._prepare_players(players)
        ball = self._prepare_ball(ball)

        timeline = self._create_timeline(players, ball)
        summary = self._create_summary(timeline)

        timeline_path = Path(timeline_output_csv)
        summary_path = Path(summary_output_csv)
        report_path = Path(report_output_json)

        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        timeline.to_csv(timeline_path, index=False)
        summary.to_csv(summary_path, index=False)

        report = self._create_report(
            players=players,
            ball=ball,
            timeline=timeline,
            summary=summary,
            players_path=players_path,
            ball_path=ball_path,
            timeline_path=timeline_path,
            summary_path=summary_path,
        )

        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return timeline, summary

    @staticmethod
    def _validate_columns(
        data: pd.DataFrame,
        required_columns: set[str],
        dataset_name: str,
    ) -> None:
        missing = required_columns.difference(data.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {dataset_name}: "
                + ", ".join(sorted(missing))
            )

    def _prepare_players(
        self,
        players: pd.DataFrame,
    ) -> pd.DataFrame:
        data = players.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "pitch_x_smooth_m",
            "pitch_y_smooth_m",
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data["class_name"] = (
            data["class_name"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        data["team_id"] = (
            data["team_id"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data = data[
            (data["class_name"] == "player")
            & data["team_id"].isin(self.VALID_TEAMS)
        ].copy()

        data = data.dropna(
            subset=[
                "frame_index",
                "time_seconds",
                "track_id",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
            ]
        )

        valid_boxes = (
            (data["bbox_x2"] > data["bbox_x1"])
            & (data["bbox_y2"] > data["bbox_y1"])
        )
        data = data[valid_boxes].copy()

        data["frame_index"] = data["frame_index"].astype(int)
        data["track_id"] = data["track_id"].astype(int)

        data["player_foot_x_px"] = (
            data["bbox_x1"] + data["bbox_x2"]
        ) / 2.0
        data["player_foot_y_px"] = data["bbox_y2"]
        data["player_height_px"] = (
            data["bbox_y2"] - data["bbox_y1"]
        ).clip(lower=self.minimum_player_height_px)

        return data.sort_values(
            ["frame_index", "track_id"]
        ).reset_index(drop=True)

    @staticmethod
    def _prepare_ball(
        ball: pd.DataFrame,
    ) -> pd.DataFrame:
        data = ball.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "pitch_x_m",
            "pitch_y_m",
            "confidence",
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = data.dropna(
            subset=[
                "frame_index",
                "time_seconds",
                "track_id",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "confidence",
            ]
        )

        valid_boxes = (
            (data["bbox_x2"] > data["bbox_x1"])
            & (data["bbox_y2"] > data["bbox_y1"])
        )
        data = data[valid_boxes].copy()

        data["frame_index"] = data["frame_index"].astype(int)
        data["track_id"] = data["track_id"].astype(int)
        data = data[data["confidence"].between(0.0, 1.0)].copy()

        data["ball_center_x_px"] = (
            data["bbox_x1"] + data["bbox_x2"]
        ) / 2.0
        data["ball_center_y_px"] = (
            data["bbox_y1"] + data["bbox_y2"]
        ) / 2.0

        # Keep the highest-confidence ball candidate per frame.
        return (
            data.sort_values(
                ["frame_index", "confidence"],
                ascending=[True, False],
            )
            .drop_duplicates(
                subset=["frame_index"],
                keep="first",
            )
            .sort_values("frame_index")
            .reset_index(drop=True)
        )

    def _create_timeline(
        self,
        players: pd.DataFrame,
        ball: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        players_by_frame = {
            int(frame_index): frame.copy()
            for frame_index, frame in players.groupby("frame_index")
        }

        current_team = "unknown"
        current_owner_id: int | None = None
        pending_team = "unknown"
        pending_owner_id: int | None = None
        pending_count = 0
        missing_control_count = 0

        for ball_row in ball.itertuples(index=False):
            frame_index = int(ball_row.frame_index)
            time_seconds = float(ball_row.time_seconds)
            ball_x_px = float(ball_row.ball_center_x_px)
            ball_y_px = float(ball_row.ball_center_y_px)

            frame_players = players_by_frame.get(frame_index)

            nearest_player_id: int | None = None
            nearest_team = "unknown"
            nearest_pixel_distance = np.nan
            nearest_normalized_distance = np.nan
            nearest_player_height = np.nan

            if frame_players is not None and not frame_players.empty:
                dx = (
                    frame_players["player_foot_x_px"].to_numpy(dtype=float)
                    - ball_x_px
                )
                dy = (
                    frame_players["player_foot_y_px"].to_numpy(dtype=float)
                    - ball_y_px
                )
                pixel_distances = np.sqrt(dx**2 + dy**2)
                player_heights = frame_players[
                    "player_height_px"
                ].to_numpy(dtype=float)
                normalized_distances = (
                    pixel_distances / player_heights
                )

                nearest_index = int(np.argmin(normalized_distances))
                nearest_record = frame_players.iloc[nearest_index]

                nearest_player_id = int(nearest_record["track_id"])
                nearest_team = str(nearest_record["team_id"])
                nearest_pixel_distance = float(
                    pixel_distances[nearest_index]
                )
                nearest_normalized_distance = float(
                    normalized_distances[nearest_index]
                )
                nearest_player_height = float(
                    player_heights[nearest_index]
                )

            candidate_team = "unknown"
            candidate_owner_id: int | None = None

            if (
                nearest_player_id is not None
                and nearest_team in self.VALID_TEAMS
                and not np.isnan(nearest_normalized_distance)
                and nearest_normalized_distance
                <= self.normalized_control_threshold
            ):
                candidate_team = nearest_team
                candidate_owner_id = nearest_player_id

            possession_state = "unknown"

            if candidate_team == "unknown":
                missing_control_count += 1
                pending_team = "unknown"
                pending_owner_id = None
                pending_count = 0

                if (
                    current_team in self.VALID_TEAMS
                    and missing_control_count <= self.hold_last_team_frames
                ):
                    possession_state = "held"
                else:
                    current_team = "unknown"
                    current_owner_id = None
                    possession_state = "unknown"

            elif candidate_team == current_team:
                missing_control_count = 0
                pending_team = "unknown"
                pending_owner_id = None
                pending_count = 0
                current_owner_id = candidate_owner_id
                possession_state = "locked"

            else:
                missing_control_count = 0

                if candidate_team == pending_team:
                    pending_count += 1
                    pending_owner_id = candidate_owner_id
                else:
                    pending_team = candidate_team
                    pending_owner_id = candidate_owner_id
                    pending_count = 1

                if pending_count >= self.switch_confirmation_frames:
                    current_team = pending_team
                    current_owner_id = pending_owner_id
                    pending_team = "unknown"
                    pending_owner_id = None
                    pending_count = 0
                    possession_state = "switched"
                elif current_team in self.VALID_TEAMS:
                    possession_state = "switching"
                else:
                    possession_state = "pending"

            rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": round(time_seconds, 3),
                    "ball_track_id": int(ball_row.track_id),
                    "ball_center_x_px": round(ball_x_px, 3),
                    "ball_center_y_px": round(ball_y_px, 3),
                    "ball_pitch_x_m": self._round_or_nan(
                        ball_row.pitch_x_m
                    ),
                    "ball_pitch_y_m": self._round_or_nan(
                        ball_row.pitch_y_m
                    ),
                    "ball_confidence": round(
                        float(ball_row.confidence),
                        4,
                    ),
                    "nearest_player_id": nearest_player_id,
                    "nearest_team": nearest_team,
                    "nearest_pixel_distance": self._round_or_nan(
                        nearest_pixel_distance
                    ),
                    "nearest_player_height_px": self._round_or_nan(
                        nearest_player_height
                    ),
                    "nearest_normalized_distance": self._round_or_nan(
                        nearest_normalized_distance
                    ),
                    "candidate_team": candidate_team,
                    "candidate_owner_id": candidate_owner_id,
                    "possession_team": current_team,
                    "possession_owner_id": current_owner_id,
                    "possession_state": possession_state,
                    "ball_state": self._ball_state(
                        candidate_team,
                        nearest_normalized_distance,
                    ),
                    "ball_zone": self._ball_zone(
                        ball_row.pitch_x_m
                    ),
                }
            )

        timeline = pd.DataFrame(rows)
        if timeline.empty:
            return timeline

        timeline["delta_time_s"] = (
            timeline["time_seconds"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )
        timeline["possession_sequence_id"] = (
            timeline["possession_team"]
            .ne(timeline["possession_team"].shift())
            .cumsum()
        )

        return timeline

    def _create_summary(
        self,
        timeline: pd.DataFrame,
    ) -> pd.DataFrame:
        if timeline.empty:
            return pd.DataFrame()

        known = timeline[
            timeline["possession_team"].isin(self.VALID_TEAMS)
        ].copy()

        known_time = float(known["delta_time_s"].sum())
        total_time = float(timeline["delta_time_s"].sum())

        rows: list[dict] = []

        for team_id in self.VALID_TEAMS:
            team_rows = timeline[
                timeline["possession_team"] == team_id
            ].copy()

            sequence_durations = (
                team_rows.groupby("possession_sequence_id")[
                    "delta_time_s"
                ].sum()
            )

            possession_time = float(
                team_rows["delta_time_s"].sum()
            )

            zone_times = (
                team_rows.groupby("ball_zone")["delta_time_s"]
                .sum()
                .to_dict()
            )

            controlled_frames = int(
                (team_rows["ball_state"] == "controlled").sum()
            )

            rows.append(
                {
                    "team_id": team_id,
                    "possession_frames": int(len(team_rows)),
                    "controlled_ball_frames": controlled_frames,
                    "possession_time_sec": round(possession_time, 3),
                    "possession_pct_known": (
                        round(
                            100.0 * possession_time / known_time,
                            2,
                        )
                        if known_time > 0
                        else 0.0
                    ),
                    "possession_pct_all_ball_time": (
                        round(
                            100.0 * possession_time / total_time,
                            2,
                        )
                        if total_time > 0
                        else 0.0
                    ),
                    "possession_sequences": int(
                        len(sequence_durations)
                    ),
                    "longest_sequence_sec": (
                        round(float(sequence_durations.max()), 3)
                        if not sequence_durations.empty
                        else 0.0
                    ),
                    "average_sequence_sec": (
                        round(float(sequence_durations.mean()), 3)
                        if not sequence_durations.empty
                        else 0.0
                    ),
                    "defensive_third_time_sec": round(
                        float(zone_times.get("defensive_third", 0.0)),
                        3,
                    ),
                    "middle_third_time_sec": round(
                        float(zone_times.get("middle_third", 0.0)),
                        3,
                    ),
                    "attacking_third_time_sec": round(
                        float(zone_times.get("attacking_third", 0.0)),
                        3,
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _create_report(
        self,
        players: pd.DataFrame,
        ball: pd.DataFrame,
        timeline: pd.DataFrame,
        summary: pd.DataFrame,
        players_path: Path,
        ball_path: Path,
        timeline_path: Path,
        summary_path: Path,
    ) -> dict:
        if timeline.empty:
            return {
                "method": "image_space_normalized_distance",
                "player_input_file": str(players_path),
                "ball_input_file": str(ball_path),
                "player_rows": int(len(players)),
                "ball_rows": int(len(ball)),
                "timeline_rows": 0,
                "summary_rows": 0,
            }

        unknown_rows = int(
            (timeline["possession_team"] == "unknown").sum()
        )
        switches = int(
            (timeline["possession_state"] == "switched").sum()
        )
        normalized_distance = pd.to_numeric(
            timeline["nearest_normalized_distance"],
            errors="coerce",
        ).dropna()

        return {
            "method": "image_space_normalized_distance",
            "player_input_file": str(players_path),
            "ball_input_file": str(ball_path),
            "timeline_output_file": str(timeline_path),
            "summary_output_file": str(summary_path),
            "player_rows": int(len(players)),
            "ball_rows": int(len(ball)),
            "timeline_rows": int(len(timeline)),
            "summary_rows": int(len(summary)),
            "ball_frames_analysed": int(
                timeline["frame_index"].nunique()
            ),
            "known_possession_rows": int(
                len(timeline) - unknown_rows
            ),
            "unknown_possession_rows": unknown_rows,
            "unknown_possession_pct": round(
                100.0 * unknown_rows / len(timeline),
                2,
            ),
            "possession_switches": switches,
            "average_nearest_normalized_distance": (
                round(float(normalized_distance.mean()), 3)
                if not normalized_distance.empty
                else None
            ),
            "median_nearest_normalized_distance": (
                round(float(normalized_distance.median()), 3)
                if not normalized_distance.empty
                else None
            ),
            "normalized_control_threshold": (
                self.normalized_control_threshold
            ),
            "switch_confirmation_frames": (
                self.switch_confirmation_frames
            ),
            "hold_last_team_frames": (
                self.hold_last_team_frames
            ),
        }

    def _ball_state(
        self,
        candidate_team: str,
        normalized_distance: float,
    ) -> str:
        if candidate_team in self.VALID_TEAMS:
            return "controlled"
        if np.isnan(normalized_distance):
            return "unknown"
        if normalized_distance <= (
            self.normalized_control_threshold * 2.0
        ):
            return "loose_ball"
        return "free_ball"

    def _ball_zone(self, pitch_x_m: float) -> str:
        if pd.isna(pitch_x_m):
            return "unknown"

        x = float(pitch_x_m)
        first_boundary = self.pitch_length_m / 3.0
        second_boundary = 2.0 * self.pitch_length_m / 3.0

        if x < first_boundary:
            return "defensive_third"
        if x < second_boundary:
            return "middle_third"
        return "attacking_third"

    @staticmethod
    def _round_or_nan(value: float) -> float:
        if pd.isna(value):
            return np.nan
        return round(float(value), 3)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate possession with image-space "
            "player-ball association."
        )
    )

    parser.add_argument(
        "--players",
        default="outputs/analytics/cleaned_tracks.csv",
        help="Path to cleaned player tracking CSV.",
    )
    parser.add_argument(
        "--ball",
        required=True,
        help="Path to ball.csv from the same run.",
    )
    parser.add_argument(
        "--timeline-output",
        default=(
            "outputs/analytics/"
            "possession_timeline_v2.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/analytics/"
            "possession_summary_v2.csv"
        ),
    )
    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "possession_report_v2.json"
        ),
    )
    parser.add_argument(
        "--normalized-threshold",
        type=float,
        default=1.5,
        help=(
            "Maximum ball-to-foot distance divided "
            "by player bounding-box height."
        ),
    )
    parser.add_argument(
        "--switch-frames",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = PossessionAnalyticsV2(
        normalized_control_threshold=(
            args.normalized_threshold
        ),
        switch_confirmation_frames=(
            args.switch_frames
        ),
        hold_last_team_frames=args.hold_frames,
    )

    timeline, summary = analyzer.run(
        players_csv=args.players,
        ball_csv=args.ball,
        timeline_output_csv=args.timeline_output,
        summary_output_csv=args.summary_output,
        report_output_json=args.report,
    )

    print("Possession analysis V2 completed.")
    print("Timeline rows:", len(timeline))
    print("Summary rows:", len(summary))
    print(
        "Timeline output:",
        Path(args.timeline_output).resolve(),
    )
    print(
        "Summary output:",
        Path(args.summary_output).resolve(),
    )
    print(
        "Report:",
        Path(args.report).resolve(),
    )


if __name__ == "__main__":
    main()

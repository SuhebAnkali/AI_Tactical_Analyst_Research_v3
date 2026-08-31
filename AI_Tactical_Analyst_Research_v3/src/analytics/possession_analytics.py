"""Estimate football possession using player and ball tracking data."""

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
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
}

BALL_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "track_id",
    "pitch_x_m",
    "pitch_y_m",
    "confidence",
}


class PossessionAnalytics:
    """Estimate team possession using nearest-player distance and smoothing."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        control_distance_m: float = 3.0,
        switch_confirmation_frames: int = 3,
        hold_last_team_frames: int = 4,
        pitch_length_m: float = 105.0,
    ) -> None:
        if control_distance_m <= 0:
            raise ValueError(
                "control_distance_m must be greater than zero."
            )

        if switch_confirmation_frames < 1:
            raise ValueError(
                "switch_confirmation_frames must be at least 1."
            )

        if hold_last_team_frames < 0:
            raise ValueError(
                "hold_last_team_frames cannot be negative."
            )

        self.control_distance_m = float(control_distance_m)
        self.switch_confirmation_frames = int(
            switch_confirmation_frames
        )
        self.hold_last_team_frames = int(
            hold_last_team_frames
        )
        self.pitch_length_m = float(pitch_length_m)

    def run(
        self,
        players_csv: str | Path,
        ball_csv: str | Path,
        timeline_output_csv: str | Path,
        summary_output_csv: str | Path,
        report_output_json: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run possession analysis and export all results."""

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

        timeline = self._create_timeline(
            players,
            ball,
        )

        summary = self._create_summary(timeline)

        timeline_path = Path(timeline_output_csv)
        summary_path = Path(summary_output_csv)
        report_path = Path(report_output_json)

        timeline_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        timeline.to_csv(
            timeline_path,
            index=False,
        )

        summary.to_csv(
            summary_path,
            index=False,
        )

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
        missing = required_columns.difference(
            data.columns
        )

        if missing:
            raise ValueError(
                f"Missing columns in {dataset_name}: "
                + ", ".join(sorted(missing))
            )

    def _prepare_players(
        self,
        players: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare team-assigned player tracking rows."""

        data = players.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
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
            & data["team_id"].isin(
                self.VALID_TEAMS
            )
        ].copy()

        data = data.dropna(
            subset=[
                "frame_index",
                "time_seconds",
                "track_id",
                "pitch_x_smooth_m",
                "pitch_y_smooth_m",
            ]
        )

        data["frame_index"] = (
            data["frame_index"].astype(int)
        )

        data["track_id"] = (
            data["track_id"].astype(int)
        )

        return data.sort_values(
            ["frame_index", "track_id"]
        ).reset_index(drop=True)

    def _prepare_ball(
        self,
        ball: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare ball data and retain one ball per frame."""

        data = ball.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
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
            subset=numeric_columns
        )

        data["frame_index"] = (
            data["frame_index"].astype(int)
        )

        data["track_id"] = (
            data["track_id"].astype(int)
        )

        data = data[
            data["confidence"].between(
                0.0,
                1.0,
            )
        ].copy()

        data = (
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

        return data

    def _create_timeline(
        self,
        players: pd.DataFrame,
        ball: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create frame-level possession estimates."""

        rows: list[dict] = []

        players_by_frame = {
            int(frame_index): frame.copy()
            for frame_index, frame
            in players.groupby("frame_index")
        }

        current_team = "unknown"
        current_owner_id: int | None = None

        pending_team = "unknown"
        pending_owner_id: int | None = None
        pending_count = 0

        missing_control_count = 0

        for ball_row in ball.itertuples(
            index=False
        ):
            frame_index = int(
                ball_row.frame_index
            )

            time_seconds = float(
                ball_row.time_seconds
            )

            ball_x = float(
                ball_row.pitch_x_m
            )

            ball_y = float(
                ball_row.pitch_y_m
            )

            nearest_player_id: int | None = None
            nearest_team = "unknown"
            nearest_distance = np.nan

            frame_players = players_by_frame.get(
                frame_index
            )

            if (
                frame_players is not None
                and not frame_players.empty
            ):
                player_x = frame_players[
                    "pitch_x_smooth_m"
                ].to_numpy(dtype=float)

                player_y = frame_players[
                    "pitch_y_smooth_m"
                ].to_numpy(dtype=float)

                distances = np.sqrt(
                    (player_x - ball_x) ** 2
                    + (player_y - ball_y) ** 2
                )

                nearest_index = int(
                    np.argmin(distances)
                )

                nearest_distance = float(
                    distances[nearest_index]
                )

                nearest_record = (
                    frame_players.iloc[
                        nearest_index
                    ]
                )

                nearest_player_id = int(
                    nearest_record["track_id"]
                )

                nearest_team = str(
                    nearest_record["team_id"]
                )

            candidate_team = "unknown"
            candidate_owner_id: int | None = None

            if (
                nearest_player_id is not None
                and nearest_team
                in self.VALID_TEAMS
                and not np.isnan(
                    nearest_distance
                )
                and nearest_distance
                <= self.control_distance_m
            ):
                candidate_team = nearest_team
                candidate_owner_id = (
                    nearest_player_id
                )

            possession_state = "unknown"

            if candidate_team == "unknown":
                missing_control_count += 1

                pending_team = "unknown"
                pending_owner_id = None
                pending_count = 0

                if (
                    current_team
                    in self.VALID_TEAMS
                    and missing_control_count
                    <= self.hold_last_team_frames
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

                current_owner_id = (
                    candidate_owner_id
                )

                possession_state = "locked"

            else:
                missing_control_count = 0

                if (
                    candidate_team
                    == pending_team
                    and candidate_owner_id
                    == pending_owner_id
                ):
                    pending_count += 1
                else:
                    pending_team = candidate_team
                    pending_owner_id = (
                        candidate_owner_id
                    )
                    pending_count = 1

                if (
                    pending_count
                    >= self.switch_confirmation_frames
                ):
                    current_team = pending_team
                    current_owner_id = (
                        pending_owner_id
                    )

                    pending_team = "unknown"
                    pending_owner_id = None
                    pending_count = 0

                    possession_state = "switched"

                elif (
                    current_team
                    in self.VALID_TEAMS
                ):
                    possession_state = "switching"

                else:
                    possession_state = "pending"

            rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": round(
                        time_seconds,
                        3,
                    ),
                    "ball_track_id": int(
                        ball_row.track_id
                    ),
                    "ball_x_m": round(
                        ball_x,
                        3,
                    ),
                    "ball_y_m": round(
                        ball_y,
                        3,
                    ),
                    "ball_confidence": round(
                        float(
                            ball_row.confidence
                        ),
                        4,
                    ),
                    "nearest_player_id":
                        nearest_player_id,
                    "nearest_team":
                        nearest_team,
                    "nearest_distance_m": (
                        round(
                            nearest_distance,
                            3,
                        )
                        if not np.isnan(
                            nearest_distance
                        )
                        else np.nan
                    ),
                    "candidate_team":
                        candidate_team,
                    "candidate_owner_id":
                        candidate_owner_id,
                    "possession_team":
                        current_team,
                    "possession_owner_id":
                        current_owner_id,
                    "possession_state":
                        possession_state,
                    "ball_zone":
                        self._ball_zone(ball_x),
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
            .ne(
                timeline[
                    "possession_team"
                ].shift()
            )
            .cumsum()
        )

        return timeline

    def _create_summary(
        self,
        timeline: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create team-level possession statistics."""

        if timeline.empty:
            return pd.DataFrame()

        known = timeline[
            timeline["possession_team"].isin(
                self.VALID_TEAMS
            )
        ].copy()

        known_time = float(
            known["delta_time_s"].sum()
        )

        total_time = float(
            timeline["delta_time_s"].sum()
        )

        rows: list[dict] = []

        for team_id in self.VALID_TEAMS:
            team_rows = timeline[
                timeline[
                    "possession_team"
                ] == team_id
            ].copy()

            sequence_durations = (
                team_rows.groupby(
                    "possession_sequence_id"
                )["delta_time_s"]
                .sum()
            )

            possession_time = float(
                team_rows[
                    "delta_time_s"
                ].sum()
            )

            zone_times = (
                team_rows.groupby(
                    "ball_zone"
                )["delta_time_s"]
                .sum()
                .to_dict()
            )

            rows.append(
                {
                    "team_id": team_id,
                    "possession_frames": int(
                        len(team_rows)
                    ),
                    "possession_time_sec":
                        round(
                            possession_time,
                            3,
                        ),
                    "possession_pct_known": (
                        round(
                            100.0
                            * possession_time
                            / known_time,
                            2,
                        )
                        if known_time > 0
                        else 0.0
                    ),
                    "possession_pct_all_ball_frames":
                        (
                            round(
                                100.0
                                * possession_time
                                / total_time,
                                2,
                            )
                            if total_time > 0
                            else 0.0
                        ),
                    "possession_sequences": int(
                        len(sequence_durations)
                    ),
                    "longest_sequence_sec": (
                        round(
                            float(
                                sequence_durations.max()
                            ),
                            3,
                        )
                        if not sequence_durations.empty
                        else 0.0
                    ),
                    "average_sequence_sec": (
                        round(
                            float(
                                sequence_durations.mean()
                            ),
                            3,
                        )
                        if not sequence_durations.empty
                        else 0.0
                    ),
                    "defensive_third_time_sec":
                        round(
                            float(
                                zone_times.get(
                                    "defensive_third",
                                    0.0,
                                )
                            ),
                            3,
                        ),
                    "middle_third_time_sec":
                        round(
                            float(
                                zone_times.get(
                                    "middle_third",
                                    0.0,
                                )
                            ),
                            3,
                        ),
                    "attacking_third_time_sec":
                        round(
                            float(
                                zone_times.get(
                                    "attacking_third",
                                    0.0,
                                )
                            ),
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
        """Create possession processing report."""

        if timeline.empty:
            return {
                "player_input_file":
                    str(players_path),
                "ball_input_file":
                    str(ball_path),
                "player_rows":
                    int(len(players)),
                "ball_rows":
                    int(len(ball)),
                "timeline_rows": 0,
                "summary_rows": 0,
            }

        unknown_rows = int(
            (
                timeline[
                    "possession_team"
                ] == "unknown"
            ).sum()
        )

        switches = int(
            (
                timeline[
                    "possession_state"
                ] == "switched"
            ).sum()
        )

        nearest_distance = pd.to_numeric(
            timeline[
                "nearest_distance_m"
            ],
            errors="coerce",
        ).dropna()

        return {
            "player_input_file":
                str(players_path),
            "ball_input_file":
                str(ball_path),
            "timeline_output_file":
                str(timeline_path),
            "summary_output_file":
                str(summary_path),
            "player_rows":
                int(len(players)),
            "ball_rows":
                int(len(ball)),
            "timeline_rows":
                int(len(timeline)),
            "summary_rows":
                int(len(summary)),
            "ball_frames_analysed":
                int(
                    timeline[
                        "frame_index"
                    ].nunique()
                ),
            "known_possession_rows":
                int(
                    len(timeline)
                    - unknown_rows
                ),
            "unknown_possession_rows":
                unknown_rows,
            "unknown_possession_pct":
                round(
                    100.0
                    * unknown_rows
                    / len(timeline),
                    2,
                ),
            "possession_switches":
                switches,
            "average_nearest_player_distance_m":
                (
                    round(
                        float(
                            nearest_distance.mean()
                        ),
                        3,
                    )
                    if not nearest_distance.empty
                    else None
                ),
            "control_distance_m":
                self.control_distance_m,
            "switch_confirmation_frames":
                self.switch_confirmation_frames,
            "hold_last_team_frames":
                self.hold_last_team_frames,
        }

    def _ball_zone(
        self,
        ball_x_m: float,
    ) -> str:
        """Return the pitch third containing the ball."""

        first_boundary = (
            self.pitch_length_m / 3.0
        )

        second_boundary = (
            2.0
            * self.pitch_length_m
            / 3.0
        )

        if ball_x_m < first_boundary:
            return "defensive_third"

        if ball_x_m < second_boundary:
            return "middle_third"

        return "attacking_third"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate football possession from "
            "player and ball tracking data."
        )
    )

    parser.add_argument(
        "--players",
        default=(
            "outputs/analytics/"
            "cleaned_tracks.csv"
        ),
    )

    parser.add_argument(
        "--ball",
        required=True,
    )

    parser.add_argument(
        "--timeline-output",
        default=(
            "outputs/analytics/"
            "possession_timeline.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/analytics/"
            "possession_summary.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "possession_report.json"
        ),
    )

    parser.add_argument(
        "--control-distance",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--switch-frames",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--hold-frames",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    analyzer = PossessionAnalytics(
        control_distance_m=(
            arguments.control_distance
        ),
        switch_confirmation_frames=(
            arguments.switch_frames
        ),
        hold_last_team_frames=(
            arguments.hold_frames
        ),
    )

    timeline, summary = analyzer.run(
        players_csv=arguments.players,
        ball_csv=arguments.ball,
        timeline_output_csv=(
            arguments.timeline_output
        ),
        summary_output_csv=(
            arguments.summary_output
        ),
        report_output_json=(
            arguments.report
        ),
    )

    print("Possession analysis completed.")
    print("Timeline rows:", len(timeline))
    print("Summary rows:", len(summary))

    print(
        "Timeline output:",
        Path(
            arguments.timeline_output
        ).resolve(),
    )

    print(
        "Summary output:",
        Path(
            arguments.summary_output
        ).resolve(),
    )

    print(
        "Report:",
        Path(arguments.report).resolve(),
    )


if __name__ == "__main__":
    main()
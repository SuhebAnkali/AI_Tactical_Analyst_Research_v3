"""Generate player-level and team-level football analytics datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "track_id",
    "class_name",
    "team_id",
    "confidence",
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
    "step_distance_m",
    "calculated_speed_mps",
    "pitch_zone",
}


class FeatureEngineeringPipeline:
    """Convert cleaned frame-level tracking data into analytics datasets."""

    def __init__(
        self,
        sprint_threshold_mps: float = 7.0,
        minimum_track_rows: int = 2,
    ) -> None:
        self.sprint_threshold_mps = sprint_threshold_mps
        self.minimum_track_rows = minimum_track_rows

    def run(
        self,
        input_csv: str | Path,
        player_output_csv: str | Path,
        team_output_csv: str | Path,
        report_path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        input_path = Path(input_csv)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Cleaned tracking file was not found: {input_path}"
            )

        data = pd.read_csv(input_path)
        self._validate_columns(data)
        data = self._prepare_data(data)

        player_statistics = self._create_player_statistics(data)
        team_statistics = self._create_team_statistics(
            data,
            player_statistics,
        )

        player_path = Path(player_output_csv)
        team_path = Path(team_output_csv)
        report_output = Path(report_path)

        player_path.parent.mkdir(parents=True, exist_ok=True)
        team_path.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)

        player_statistics.to_csv(player_path, index=False)
        team_statistics.to_csv(team_path, index=False)

        report = {
            "input_file": str(input_path),
            "input_player_rows": int(len(data)),
            "unique_player_tracks": int(data["track_id"].nunique()),
            "player_statistics_rows": int(len(player_statistics)),
            "team_statistics_rows": int(len(team_statistics)),
            "unknown_player_tracks": int(
                (player_statistics["team_id"] == "unknown").sum()
            )
            if not player_statistics.empty
            else 0,
            "player_output": str(player_path),
            "team_output": str(team_path),
        }

        report_output.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return player_statistics, team_statistics

    @staticmethod
    def _validate_columns(data: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(data.columns)

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
            "confidence",
            "pitch_x_smooth_m",
            "pitch_y_smooth_m",
            "step_distance_m",
            "calculated_speed_mps",
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

        data["pitch_zone"] = (
            data["pitch_zone"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Player statistics should not include referees.
        data = data[data["class_name"] == "player"].copy()

        data = data.dropna(
            subset=[
                "frame_index",
                "time_seconds",
                "track_id",
            ]
        )

        data["frame_index"] = data["frame_index"].astype(int)
        data["track_id"] = data["track_id"].astype(int)

        data["step_distance_m"] = (
            data["step_distance_m"]
            .fillna(0.0)
            .clip(lower=0.0)
        )

        data["calculated_speed_mps"] = (
            data["calculated_speed_mps"]
            .fillna(0.0)
            .clip(lower=0.0)
        )

        data["is_sprint"] = (
            data["calculated_speed_mps"]
            >= self.sprint_threshold_mps
        )

        return data.sort_values(
            ["track_id", "frame_index"]
        ).reset_index(drop=True)

    def _create_player_statistics(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        for track_id, track in data.groupby("track_id"):
            track = track.sort_values("frame_index")

            if len(track) < self.minimum_track_rows:
                continue

            team_id = self._select_team(track["team_id"])

            total_distance = float(
                track["step_distance_m"].sum()
            )

            average_speed = float(
                track["calculated_speed_mps"].mean()
            )

            maximum_speed = float(
                track["calculated_speed_mps"].max()
            )

            active_time = max(
                0.0,
                float(
                    track["time_seconds"].max()
                    - track["time_seconds"].min()
                ),
            )

            sprint_count = self._count_sprint_events(
                track["is_sprint"]
            )

            zone_percentages = (
                track["pitch_zone"]
                .value_counts(normalize=True)
                .mul(100.0)
            )

            valid_x = track["pitch_x_smooth_m"].dropna()
            valid_y = track["pitch_y_smooth_m"].dropna()

            rows.append(
                {
                    "track_id": int(track_id),
                    "team_id": team_id,
                    "active_frames": int(len(track)),
                    "first_frame": int(
                        track["frame_index"].min()
                    ),
                    "last_frame": int(
                        track["frame_index"].max()
                    ),
                    "active_time_sec": round(
                        active_time,
                        3,
                    ),
                    "total_distance_m": round(
                        total_distance,
                        3,
                    ),
                    "average_speed_mps": round(
                        average_speed,
                        3,
                    ),
                    "maximum_speed_mps": round(
                        maximum_speed,
                        3,
                    ),
                    "sprint_count": int(sprint_count),
                    "average_pitch_x_m": round(
                        float(valid_x.mean()),
                        3,
                    )
                    if not valid_x.empty
                    else np.nan,
                    "average_pitch_y_m": round(
                        float(valid_y.mean()),
                        3,
                    )
                    if not valid_y.empty
                    else np.nan,
                    "defensive_third_pct": round(
                        float(
                            zone_percentages.get(
                                "defensive_third",
                                0.0,
                            )
                        ),
                        2,
                    ),
                    "middle_third_pct": round(
                        float(
                            zone_percentages.get(
                                "middle_third",
                                0.0,
                            )
                        ),
                        2,
                    ),
                    "attacking_third_pct": round(
                        float(
                            zone_percentages.get(
                                "attacking_third",
                                0.0,
                            )
                        ),
                        2,
                    ),
                    "mean_detection_confidence": round(
                        float(track["confidence"].mean()),
                        4,
                    ),
                    "team_assignment_ratio": round(
                        float(
                            (
                                track["team_id"]
                                != "unknown"
                            ).mean()
                        ),
                        4,
                    ),
                    "tracking_continuity_pct": round(
                        self._tracking_continuity(track),
                        2,
                    ),
                }
            )

        return pd.DataFrame(rows).sort_values(
            ["team_id", "track_id"]
        ).reset_index(drop=True)

    def _create_team_statistics(
        self,
        frame_data: pd.DataFrame,
        player_statistics: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        for team_id in ["team_home", "team_away"]:
            players = player_statistics[
                player_statistics["team_id"] == team_id
            ]

            team_frames = frame_data[
                frame_data["team_id"] == team_id
            ]

            if players.empty:
                continue

            frame_shape = (
                team_frames.groupby("frame_index")
                .agg(
                    team_width_m=(
                        "pitch_y_smooth_m",
                        self._range_value,
                    ),
                    team_length_m=(
                        "pitch_x_smooth_m",
                        self._range_value,
                    ),
                    centroid_x_m=(
                        "pitch_x_smooth_m",
                        "mean",
                    ),
                    centroid_y_m=(
                        "pitch_y_smooth_m",
                        "mean",
                    ),
                    visible_players=(
                        "track_id",
                        "nunique",
                    ),
                )
                .reset_index()
            )

            rows.append(
                {
                    "team_id": team_id,
                    "tracked_players": int(
                        players["track_id"].nunique()
                    ),
                    "total_distance_m": round(
                        float(
                            players[
                                "total_distance_m"
                            ].sum()
                        ),
                        3,
                    ),
                    "mean_player_distance_m": round(
                        float(
                            players[
                                "total_distance_m"
                            ].mean()
                        ),
                        3,
                    ),
                    "average_speed_mps": round(
                        float(
                            players[
                                "average_speed_mps"
                            ].mean()
                        ),
                        3,
                    ),
                    "maximum_speed_mps": round(
                        float(
                            players[
                                "maximum_speed_mps"
                            ].max()
                        ),
                        3,
                    ),
                    "total_sprints": int(
                        players["sprint_count"].sum()
                    ),
                    "average_pitch_x_m": round(
                        float(
                            players[
                                "average_pitch_x_m"
                            ].mean()
                        ),
                        3,
                    ),
                    "average_pitch_y_m": round(
                        float(
                            players[
                                "average_pitch_y_m"
                            ].mean()
                        ),
                        3,
                    ),
                    "mean_team_width_m": round(
                        float(
                            frame_shape[
                                "team_width_m"
                            ].mean()
                        ),
                        3,
                    ),
                    "mean_team_length_m": round(
                        float(
                            frame_shape[
                                "team_length_m"
                            ].mean()
                        ),
                        3,
                    ),
                    "mean_visible_players_per_frame": round(
                        float(
                            frame_shape[
                                "visible_players"
                            ].mean()
                        ),
                        3,
                    ),
                    "mean_detection_confidence": round(
                        float(
                            players[
                                "mean_detection_confidence"
                            ].mean()
                        ),
                        4,
                    ),
                    "mean_tracking_continuity_pct": round(
                        float(
                            players[
                                "tracking_continuity_pct"
                            ].mean()
                        ),
                        2,
                    ),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _select_team(team_values: pd.Series) -> str:
        values = (
            team_values
            .dropna()
            .astype(str)
            .str.lower()
        )

        assigned = values[
            values.isin(["team_home", "team_away"])
        ]

        if not assigned.empty:
            mode = assigned.mode()
            if not mode.empty:
                return str(mode.iloc[0])

        return "unknown"

    @staticmethod
    def _count_sprint_events(
        sprint_values: pd.Series,
    ) -> int:
        sprint_values = (
            sprint_values
            .fillna(False)
            .astype(bool)
        )

        sprint_starts = (
            sprint_values
            & ~sprint_values.shift(
                1,
                fill_value=False,
            )
        )

        return int(sprint_starts.sum())

    @staticmethod
    def _tracking_continuity(
        track: pd.DataFrame,
    ) -> float:
        first_frame = int(track["frame_index"].min())
        last_frame = int(track["frame_index"].max())

        expected_frames = (
            last_frame - first_frame + 1
        )

        if expected_frames <= 0:
            return 0.0

        return (
            100.0
            * len(track)
            / expected_frames
        )

    @staticmethod
    def _range_value(
        values: pd.Series,
    ) -> float:
        values = values.dropna()

        if values.empty:
            return 0.0

        return float(
            values.max() - values.min()
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate player and team football "
            "analytics datasets."
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "outputs/analytics/"
            "cleaned_tracks.csv"
        ),
    )

    parser.add_argument(
        "--player-output",
        default=(
            "outputs/analytics/"
            "player_statistics.csv"
        ),
    )

    parser.add_argument(
        "--team-output",
        default=(
            "outputs/analytics/"
            "team_statistics.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "feature_engineering_report.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    pipeline = FeatureEngineeringPipeline()

    player_statistics, team_statistics = pipeline.run(
        input_csv=arguments.input,
        player_output_csv=arguments.player_output,
        team_output_csv=arguments.team_output,
        report_path=arguments.report,
    )

    print("Feature engineering completed.")
    print(
        "Player rows:",
        len(player_statistics),
    )
    print(
        "Team rows:",
        len(team_statistics),
    )
    print(
        "Player output:",
        Path(arguments.player_output).resolve(),
    )
    print(
        "Team output:",
        Path(arguments.team_output).resolve(),
    )
    print(
        "Report:",
        Path(arguments.report).resolve(),
    )


if __name__ == "__main__":
    main()
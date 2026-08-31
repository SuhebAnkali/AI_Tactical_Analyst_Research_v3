"""Generate frame-by-frame tactical analytics."""

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
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
    "calculated_speed_mps",
}


class TacticalFrameAnalytics:
    """Calculate team shape and spatial metrics for every frame."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        minimum_players: int = 3,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.minimum_players = minimum_players

    def run(
        self,
        input_csv: str | Path,
        timeline_output: str | Path,
        summary_output: str | Path,
        report_output: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        input_path = Path(input_csv)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Cleaned tracks file not found: {input_path}"
            )

        data = pd.read_csv(input_path)

        self._validate_columns(data)

        data = self._prepare(data)

        timeline = self._create_timeline(data)
        summary = self._create_summary(timeline)

        timeline_path = Path(timeline_output)
        summary_path = Path(summary_output)
        report_path = Path(report_output)

        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        timeline.to_csv(timeline_path, index=False)
        summary.to_csv(summary_path, index=False)

        report = {
            "input_file": str(input_path),
            "input_rows": int(len(data)),
            "frames_analysed": int(len(timeline)),
            "timeline_rows": int(len(timeline)),
            "summary_rows": int(len(summary)),
            "timeline_output": str(timeline_path),
            "summary_output": str(summary_path),
        }

        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return timeline, summary

    @staticmethod
    def _validate_columns(data: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(data.columns)

        if missing:
            raise ValueError(
                "Missing columns: "
                + ", ".join(sorted(missing))
            )

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "track_id",
            "pitch_x_smooth_m",
            "pitch_y_smooth_m",
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

        data = data[
            (data["class_name"] == "player")
            & data["team_id"].isin(self.VALID_TEAMS)
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

        data["frame_index"] = data["frame_index"].astype(int)
        data["track_id"] = data["track_id"].astype(int)

        data["pitch_x_smooth_m"] = data[
            "pitch_x_smooth_m"
        ].clip(0.0, self.pitch_length_m)

        data["pitch_y_smooth_m"] = data[
            "pitch_y_smooth_m"
        ].clip(0.0, self.pitch_width_m)

        return data.sort_values(
            ["frame_index", "team_id", "track_id"]
        ).reset_index(drop=True)

    def _create_timeline(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        rows: list[dict] = []

        for frame_index, frame in data.groupby("frame_index"):
            home = frame[
                frame["team_id"] == "team_home"
            ]

            away = frame[
                frame["team_id"] == "team_away"
            ]

            home_metrics = self._team_metrics(home)
            away_metrics = self._team_metrics(away)

            rows.append(
                {
                    "frame_index": int(frame_index),
                    "time_seconds": round(
                        float(frame["time_seconds"].median()),
                        3,
                    ),

                    **self._prefix("home", home_metrics),
                    **self._prefix("away", away_metrics),

                    "centroid_distance_m":
                        self._point_distance(
                            home_metrics["centroid_x_m"],
                            home_metrics["centroid_y_m"],
                            away_metrics["centroid_x_m"],
                            away_metrics["centroid_y_m"],
                        ),

                    "horizontal_separation_m":
                        self._difference(
                            home_metrics["centroid_x_m"],
                            away_metrics["centroid_x_m"],
                        ),

                    "vertical_separation_m":
                        self._difference(
                            home_metrics["centroid_y_m"],
                            away_metrics["centroid_y_m"],
                        ),

                    "numerical_balance_home_minus_away":
                        int(
                            home_metrics["visible_players"]
                            - away_metrics["visible_players"]
                        ),

                    "home_valid_shape":
                        home_metrics["visible_players"]
                        >= self.minimum_players,

                    "away_valid_shape":
                        away_metrics["visible_players"]
                        >= self.minimum_players,
                }
            )

        return pd.DataFrame(rows).sort_values(
            "frame_index"
        ).reset_index(drop=True)

    def _team_metrics(
        self,
        team_frame: pd.DataFrame,
    ) -> dict:

        if team_frame.empty:
            return {
                "visible_players": 0,
                "centroid_x_m": np.nan,
                "centroid_y_m": np.nan,
                "width_m": np.nan,
                "length_m": np.nan,
                "compactness_m": np.nan,
                "defensive_line_x_m": np.nan,
                "attacking_line_x_m": np.nan,
                "mean_speed_mps": np.nan,
                "pitch_occupation_pct": np.nan,
            }

        x = team_frame["pitch_x_smooth_m"].to_numpy(
            dtype=float
        )

        y = team_frame["pitch_y_smooth_m"].to_numpy(
            dtype=float
        )

        centroid_x = float(np.mean(x))
        centroid_y = float(np.mean(y))

        width = float(np.max(y) - np.min(y))
        length = float(np.max(x) - np.min(x))

        compactness = float(
            np.mean(
                np.sqrt(
                    (x - centroid_x) ** 2
                    + (y - centroid_y) ** 2
                )
            )
        )

        defensive_line = float(np.percentile(x, 20))
        attacking_line = float(np.percentile(x, 80))

        speed = pd.to_numeric(
            team_frame["calculated_speed_mps"],
            errors="coerce",
        ).dropna()

        mean_speed = (
            float(speed.mean())
            if not speed.empty
            else np.nan
        )

        pitch_area = (
            self.pitch_length_m
            * self.pitch_width_m
        )

        occupied_area = width * length

        occupation_pct = (
            100.0 * occupied_area / pitch_area
            if pitch_area > 0
            else np.nan
        )

        return {
            "visible_players": int(
                team_frame["track_id"].nunique()
            ),
            "centroid_x_m": round(centroid_x, 3),
            "centroid_y_m": round(centroid_y, 3),
            "width_m": round(width, 3),
            "length_m": round(length, 3),
            "compactness_m": round(compactness, 3),
            "defensive_line_x_m": round(
                defensive_line,
                3,
            ),
            "attacking_line_x_m": round(
                attacking_line,
                3,
            ),
            "mean_speed_mps": (
                round(mean_speed, 3)
                if not np.isnan(mean_speed)
                else np.nan
            ),
            "pitch_occupation_pct": round(
                occupation_pct,
                3,
            ),
        }

    @staticmethod
    def _prefix(prefix: str, metrics: dict) -> dict:
        return {
            f"{prefix}_{key}": value
            for key, value in metrics.items()
        }

    def _create_summary(
        self,
        timeline: pd.DataFrame,
    ) -> pd.DataFrame:

        rows: list[dict] = []

        for team_id, prefix in [
            ("team_home", "home"),
            ("team_away", "away"),
        ]:
            valid = timeline[
                timeline[f"{prefix}_valid_shape"] == True
            ].copy()

            if valid.empty:
                valid = timeline.copy()

            rows.append(
                {
                    "team_id": team_id,
                    "frames_analysed": int(len(timeline)),
                    "valid_shape_frames": int(
                        timeline[
                            f"{prefix}_valid_shape"
                        ].sum()
                    ),
                    "mean_visible_players":
                        self._mean(
                            valid[
                                f"{prefix}_visible_players"
                            ]
                        ),
                    "mean_centroid_x_m":
                        self._mean(
                            valid[
                                f"{prefix}_centroid_x_m"
                            ]
                        ),
                    "mean_centroid_y_m":
                        self._mean(
                            valid[
                                f"{prefix}_centroid_y_m"
                            ]
                        ),
                    "mean_width_m":
                        self._mean(
                            valid[
                                f"{prefix}_width_m"
                            ]
                        ),
                    "mean_length_m":
                        self._mean(
                            valid[
                                f"{prefix}_length_m"
                            ]
                        ),
                    "mean_compactness_m":
                        self._mean(
                            valid[
                                f"{prefix}_compactness_m"
                            ]
                        ),
                    "mean_defensive_line_x_m":
                        self._mean(
                            valid[
                                f"{prefix}_defensive_line_x_m"
                            ]
                        ),
                    "mean_attacking_line_x_m":
                        self._mean(
                            valid[
                                f"{prefix}_attacking_line_x_m"
                            ]
                        ),
                    "mean_speed_mps":
                        self._mean(
                            valid[
                                f"{prefix}_mean_speed_mps"
                            ]
                        ),
                    "mean_pitch_occupation_pct":
                        self._mean(
                            valid[
                                f"{prefix}_pitch_occupation_pct"
                            ]
                        ),
                    "mean_centroid_distance_m":
                        self._mean(
                            valid[
                                "centroid_distance_m"
                            ]
                        ),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _point_distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:

        if any(
            pd.isna(value)
            for value in [x1, y1, x2, y2]
        ):
            return np.nan

        return round(
            float(
                np.hypot(
                    x1 - x2,
                    y1 - y2,
                )
            ),
            3,
        )

    @staticmethod
    def _difference(
        value_a: float,
        value_b: float,
    ) -> float:

        if pd.isna(value_a) or pd.isna(value_b):
            return np.nan

        return round(
            abs(float(value_a) - float(value_b)),
            3,
        )

    @staticmethod
    def _mean(series: pd.Series) -> float:
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return round(float(values.mean()), 3)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate tactical frame analytics."
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
        "--timeline-output",
        default=(
            "outputs/analytics/"
            "tactical_timeline.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/analytics/"
            "tactical_summary.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "tactical_analytics_report.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = TacticalFrameAnalytics()

    timeline, summary = analyzer.run(
        input_csv=args.input,
        timeline_output=args.timeline_output,
        summary_output=args.summary_output,
        report_output=args.report,
    )

    print("Tactical frame analytics completed.")
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
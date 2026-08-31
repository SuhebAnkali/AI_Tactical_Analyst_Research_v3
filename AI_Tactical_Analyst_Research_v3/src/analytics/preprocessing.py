"""Preprocess raw football tracking data for tactical analytics."""

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
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pitch_x_m",
    "pitch_y_m",
    "speed_mps",
    "confidence",
}

NUMERIC_COLUMNS = [
    "frame_index",
    "time_seconds",
    "track_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "pitch_x_m",
    "pitch_y_m",
    "speed_mps",
    "confidence",
]


class TrackingPreprocessor:
    """Clean and transform raw object-tracking records."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        maximum_player_speed: float = 12.5,
    ) -> None:
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.maximum_player_speed = maximum_player_speed

    def process(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        report_path: str | Path,
    ) -> pd.DataFrame:
        """Run the complete preprocessing workflow."""

        input_path = Path(input_csv)
        output_path = Path(output_csv)
        report_output = Path(report_path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Tracking CSV was not found: {input_path}"
            )

        dataframe = pd.read_csv(input_path)

        self._validate_columns(dataframe)

        original_rows = len(dataframe)
        original_tracks = dataframe["track_id"].nunique()

        dataframe = self._convert_types(dataframe)
        dataframe = self._remove_invalid_rows(dataframe)
        dataframe = self._remove_duplicates(dataframe)
        dataframe = self._validate_pitch_coordinates(dataframe)
        dataframe = self._validate_speed(dataframe)
        dataframe = self._interpolate_coordinates(dataframe)
        dataframe = self._smooth_coordinates(dataframe)
        dataframe = self._recalculate_movement(dataframe)
        dataframe = self._add_engineered_columns(dataframe)

        dataframe = dataframe.sort_values(
            ["track_id", "frame_index"]
        ).reset_index(drop=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)

        report = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "original_rows": int(original_rows),
            "cleaned_rows": int(len(dataframe)),
            "removed_rows": int(original_rows - len(dataframe)),
            "original_unique_tracks": int(original_tracks),
            "cleaned_unique_tracks": int(
                dataframe["track_id"].nunique()
            ),
            "unknown_team_rows": int(
                (dataframe["team_id"] == "unknown").sum()
            ),
            "missing_pitch_coordinate_rows": int(
                dataframe[
                    ["pitch_x_m", "pitch_y_m"]
                ].isna().any(axis=1).sum()
            ),
            "class_counts": {
                str(key): int(value)
                for key, value in dataframe[
                    "class_name"
                ].value_counts().items()
            },
            "team_counts": {
                str(key): int(value)
                for key, value in dataframe[
                    "team_id"
                ].value_counts().items()
            },
        }

        report_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        report_output.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return dataframe

    @staticmethod
    def _validate_columns(dataframe: pd.DataFrame) -> None:
        missing = REQUIRED_COLUMNS.difference(
            dataframe.columns
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _convert_types(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        for column in NUMERIC_COLUMNS:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

        dataframe["class_name"] = (
            dataframe["class_name"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        dataframe["team_id"] = (
            dataframe["team_id"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        return dataframe

    @staticmethod
    def _remove_invalid_rows(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.dropna(
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
        ).copy()

        valid_boxes = (
            (dataframe["bbox_x2"] > dataframe["bbox_x1"])
            & (
                dataframe["bbox_y2"]
                > dataframe["bbox_y1"]
            )
        )

        valid_confidence = dataframe[
            "confidence"
        ].between(0.0, 1.0)

        valid_identifiers = (
            (dataframe["frame_index"] >= 0)
            & (dataframe["track_id"] >= 0)
        )

        dataframe = dataframe[
            valid_boxes
            & valid_confidence
            & valid_identifiers
        ].copy()

        dataframe["frame_index"] = dataframe[
            "frame_index"
        ].astype(int)

        dataframe["track_id"] = dataframe[
            "track_id"
        ].astype(int)

        return dataframe

    @staticmethod
    def _remove_duplicates(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        return dataframe.drop_duplicates(
            subset=[
                "frame_index",
                "track_id",
                "class_name",
            ],
            keep="last",
        ).copy()

    def _validate_pitch_coordinates(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        valid_x = dataframe["pitch_x_m"].between(
            0.0,
            self.pitch_length,
        )

        valid_y = dataframe["pitch_y_m"].between(
            0.0,
            self.pitch_width,
        )

        invalid_coordinates = ~(valid_x & valid_y)

        dataframe.loc[
            invalid_coordinates,
            ["pitch_x_m", "pitch_y_m"],
        ] = np.nan

        return dataframe

    def _validate_speed(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        invalid_speed = (
            (dataframe["speed_mps"] < 0)
            | (
                dataframe["speed_mps"]
                > self.maximum_player_speed
            )
        )

        dataframe.loc[
            invalid_speed,
            "speed_mps",
        ] = np.nan

        return dataframe

    @staticmethod
    def _interpolate_coordinates(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.sort_values(
            ["track_id", "frame_index"]
        ).copy()

        for column in ["pitch_x_m", "pitch_y_m"]:
            dataframe[column] = dataframe.groupby(
                "track_id"
            )[column].transform(
                lambda series: series.interpolate(
                    method="linear",
                    limit=3,
                    limit_direction="both",
                )
            )

        return dataframe

    @staticmethod
    def _smooth_coordinates(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        dataframe["pitch_x_smooth_m"] = (
            dataframe.groupby("track_id")[
                "pitch_x_m"
            ].transform(
                lambda series: series.rolling(
                    window=5,
                    min_periods=1,
                    center=True,
                ).median()
            )
        )

        dataframe["pitch_y_smooth_m"] = (
            dataframe.groupby("track_id")[
                "pitch_y_m"
            ].transform(
                lambda series: series.rolling(
                    window=5,
                    min_periods=1,
                    center=True,
                ).median()
            )
        )

        return dataframe

    def _recalculate_movement(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.sort_values(
            ["track_id", "frame_index"]
        ).copy()

        grouped = dataframe.groupby(
            "track_id",
            group_keys=False,
        )

        dataframe["delta_x_m"] = grouped[
            "pitch_x_smooth_m"
        ].diff()

        dataframe["delta_y_m"] = grouped[
            "pitch_y_smooth_m"
        ].diff()

        dataframe["delta_time_s"] = grouped[
            "time_seconds"
        ].diff()

        dataframe["step_distance_m"] = np.sqrt(
            dataframe["delta_x_m"].pow(2)
            + dataframe["delta_y_m"].pow(2)
        )

        invalid_time = dataframe["delta_time_s"] <= 0

        dataframe.loc[
            invalid_time,
            [
                "delta_time_s",
                "step_distance_m",
            ],
        ] = np.nan

        dataframe["calculated_speed_mps"] = (
            dataframe["step_distance_m"]
            / dataframe["delta_time_s"]
        )

        impossible_movement = (
            dataframe["calculated_speed_mps"]
            > self.maximum_player_speed
        )

        dataframe.loc[
            impossible_movement,
            [
                "step_distance_m",
                "calculated_speed_mps",
            ],
        ] = np.nan

        dataframe["calculated_speed_mps"] = (
            dataframe.groupby("track_id")[
                "calculated_speed_mps"
            ].transform(
                lambda series: series.rolling(
                    window=5,
                    min_periods=1,
                ).median()
            )
        )

        return dataframe

    @staticmethod
    def _add_engineered_columns(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        dataframe = dataframe.copy()

        dataframe["bbox_width_px"] = (
            dataframe["bbox_x2"]
            - dataframe["bbox_x1"]
        )

        dataframe["bbox_height_px"] = (
            dataframe["bbox_y2"]
            - dataframe["bbox_y1"]
        )

        dataframe["bbox_area_px"] = (
            dataframe["bbox_width_px"]
            * dataframe["bbox_height_px"]
        )

        dataframe["bbox_center_x_px"] = (
            dataframe["bbox_x1"]
            + dataframe["bbox_x2"]
        ) / 2.0

        dataframe["bbox_center_y_px"] = (
            dataframe["bbox_y1"]
            + dataframe["bbox_y2"]
        ) / 2.0

        dataframe["is_team_assigned"] = (
            dataframe["team_id"].isin(
                ["team_home", "team_away"]
            )
        )

        dataframe["is_sprint"] = (
            dataframe["calculated_speed_mps"] >= 7.0
        )

        dataframe["pitch_zone"] = pd.cut(
            dataframe["pitch_x_smooth_m"],
            bins=[-np.inf, 35.0, 70.0, np.inf],
            labels=[
                "defensive_third",
                "middle_third",
                "attacking_third",
            ],
        )

        return dataframe


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean raw football tracking CSV data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw tracks.csv",
    )

    parser.add_argument(
        "--output",
        default="outputs/analytics/cleaned_tracks.csv",
        help="Path for cleaned tracking dataset",
    )

    parser.add_argument(
        "--report",
        default="outputs/analytics/preprocessing_report.json",
        help="Path for preprocessing report",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    preprocessor = TrackingPreprocessor()

    cleaned = preprocessor.process(
        input_csv=arguments.input,
        output_csv=arguments.output,
        report_path=arguments.report,
    )

    print("Preprocessing completed.")
    print(f"Cleaned rows: {len(cleaned)}")
    print(
        "Unique tracks:",
        cleaned["track_id"].nunique(),
    )
    print(
        "Output:",
        Path(arguments.output).resolve(),
    )
    print(
        "Report:",
        Path(arguments.report).resolve(),
    )


if __name__ == "__main__":
    main()
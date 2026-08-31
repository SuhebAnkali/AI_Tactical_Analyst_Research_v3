"""Short-sequence football transition analysis.

This module is designed for short broadcast clips such as ``clean_match1.mp4``.
It does not claim to recover a full-match tactical formation. Instead, it
estimates the observed transition phases and team-shape changes during the clip.

Inputs
------
- cleaned_tracks.csv
- tactical_timeline.csv
- possession_timeline_v2.csv

Outputs
-------
- transition_timeline.csv
- transition_summary.csv
- transition_report.json

Main metrics
------------
- attacking team estimate
- attack direction
- team centroid progression
- team width, length and compactness
- final-third entry
- attacking tempo proxy
- overload side
- transition phase
- confidence and data-quality flags
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRACK_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "track_id",
    "class_name",
    "team_id",
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
    "calculated_speed_mps",
}

TACTICAL_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "home_visible_players",
    "home_centroid_x_m",
    "home_centroid_y_m",
    "home_width_m",
    "home_length_m",
    "home_compactness_m",
    "away_visible_players",
    "away_centroid_x_m",
    "away_centroid_y_m",
    "away_width_m",
    "away_length_m",
    "away_compactness_m",
}

POSSESSION_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "possession_team",
    "possession_state",
    "ball_pitch_x_m",
    "ball_pitch_y_m",
    "ball_zone",
}


class TransitionAnalyzer:
    """Estimate tactical transition phases from a short football clip."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        minimum_visible_players: int = 3,
        smoothing_window: int = 5,
        progression_threshold_mps: float = 1.0,
        high_tempo_threshold_mps: float = 2.5,
    ) -> None:
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)
        self.minimum_visible_players = int(minimum_visible_players)
        self.smoothing_window = max(1, int(smoothing_window))
        self.progression_threshold_mps = float(progression_threshold_mps)
        self.high_tempo_threshold_mps = float(high_tempo_threshold_mps)

    def run(
        self,
        tracks_csv: str | Path,
        tactical_csv: str | Path,
        possession_csv: str | Path,
        timeline_output_csv: str | Path,
        summary_output_csv: str | Path,
        report_output_json: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run transition analysis and export timeline, summary and report."""

        tracks_path = Path(tracks_csv)
        tactical_path = Path(tactical_csv)
        possession_path = Path(possession_csv)

        for path, label in (
            (tracks_path, "cleaned tracks"),
            (tactical_path, "tactical timeline"),
            (possession_path, "possession timeline"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        tracks = pd.read_csv(tracks_path)
        tactical = pd.read_csv(tactical_path)
        possession = pd.read_csv(possession_path)

        self._validate_columns(
            tracks,
            TRACK_REQUIRED_COLUMNS,
            "cleaned tracks",
        )
        self._validate_columns(
            tactical,
            TACTICAL_REQUIRED_COLUMNS,
            "tactical timeline",
        )
        self._validate_columns(
            possession,
            POSSESSION_REQUIRED_COLUMNS,
            "possession timeline",
        )

        tracks = self._prepare_tracks(tracks)
        tactical = self._prepare_tactical(tactical)
        possession = self._prepare_possession(possession)

        timeline = self._build_timeline(
            tracks=tracks,
            tactical=tactical,
            possession=possession,
        )
        summary = self._build_summary(timeline)

        timeline_path = Path(timeline_output_csv)
        summary_path = Path(summary_output_csv)
        report_path = Path(report_output_json)

        timeline_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        timeline.to_csv(timeline_path, index=False)
        summary.to_csv(summary_path, index=False)

        report = self._build_report(
            tracks=tracks,
            tactical=tactical,
            possession=possession,
            timeline=timeline,
            summary=summary,
            tracks_path=tracks_path,
            tactical_path=tactical_path,
            possession_path=possession_path,
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
        required: set[str],
        dataset_name: str,
    ) -> None:
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {dataset_name}: "
                + ", ".join(sorted(missing))
            )

    def _prepare_tracks(self, tracks: pd.DataFrame) -> pd.DataFrame:
        data = tracks.copy()

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

        return data.sort_values(
            ["frame_index", "team_id", "track_id"]
        ).reset_index(drop=True)

    def _prepare_tactical(self, tactical: pd.DataFrame) -> pd.DataFrame:
        data = tactical.copy()

        for column in data.columns:
            if column not in {"home_valid_shape", "away_valid_shape"}:
                data[column] = pd.to_numeric(
                    data[column],
                    errors="coerce",
                )

        data["frame_index"] = data["frame_index"].astype(int)

        return data.sort_values("frame_index").reset_index(drop=True)

    def _prepare_possession(self, possession: pd.DataFrame) -> pd.DataFrame:
        data = possession.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "ball_pitch_x_m",
            "ball_pitch_y_m",
        ]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data["possession_team"] = (
            data["possession_team"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        data["possession_state"] = (
            data["possession_state"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        data["ball_zone"] = (
            data["ball_zone"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        data = data.dropna(subset=["frame_index", "time_seconds"])
        data["frame_index"] = data["frame_index"].astype(int)

        return data.sort_values("frame_index").reset_index(drop=True)

    def _build_timeline(
        self,
        tracks: pd.DataFrame,
        tactical: pd.DataFrame,
        possession: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create frame-level transition analytics."""

        merged = tactical.merge(
            possession[
                [
                    "frame_index",
                    "possession_team",
                    "possession_state",
                    "ball_pitch_x_m",
                    "ball_pitch_y_m",
                    "ball_zone",
                ]
            ],
            on="frame_index",
            how="left",
        )

        merged["possession_team"] = merged[
            "possession_team"
        ].fillna("unknown")
        merged["possession_state"] = merged[
            "possession_state"
        ].fillna("unknown")
        merged["ball_zone"] = merged["ball_zone"].fillna("unknown")

        frame_speed = (
            tracks.groupby(["frame_index", "team_id"])
            .agg(
                team_mean_speed_mps=(
                    "calculated_speed_mps",
                    "mean",
                ),
                team_max_speed_mps=(
                    "calculated_speed_mps",
                    "max",
                ),
                visible_players=("track_id", "nunique"),
            )
            .reset_index()
        )

        home_speed = frame_speed[
            frame_speed["team_id"] == "team_home"
        ][
            [
                "frame_index",
                "team_mean_speed_mps",
                "team_max_speed_mps",
            ]
        ].rename(
            columns={
                "team_mean_speed_mps": "home_mean_speed_from_tracks_mps",
                "team_max_speed_mps": "home_max_speed_mps",
            }
        )

        away_speed = frame_speed[
            frame_speed["team_id"] == "team_away"
        ][
            [
                "frame_index",
                "team_mean_speed_mps",
                "team_max_speed_mps",
            ]
        ].rename(
            columns={
                "team_mean_speed_mps": "away_mean_speed_from_tracks_mps",
                "team_max_speed_mps": "away_max_speed_mps",
            }
        )

        merged = merged.merge(
            home_speed,
            on="frame_index",
            how="left",
        ).merge(
            away_speed,
            on="frame_index",
            how="left",
        )

        merged = merged.sort_values("frame_index").reset_index(drop=True)

        merged["delta_time_s"] = (
            merged["time_seconds"]
            .diff()
            .replace(0, np.nan)
        )

        for prefix in ("home", "away"):
            merged[f"{prefix}_centroid_dx_m"] = (
                merged[f"{prefix}_centroid_x_m"].diff()
            )
            merged[f"{prefix}_centroid_dy_m"] = (
                merged[f"{prefix}_centroid_y_m"].diff()
            )
            merged[f"{prefix}_centroid_speed_mps"] = (
                np.sqrt(
                    merged[f"{prefix}_centroid_dx_m"] ** 2
                    + merged[f"{prefix}_centroid_dy_m"] ** 2
                )
                / merged["delta_time_s"]
            )

            merged[f"{prefix}_centroid_x_smooth_m"] = (
                merged[f"{prefix}_centroid_x_m"]
                .rolling(
                    window=self.smoothing_window,
                    min_periods=1,
                    center=True,
                )
                .median()
            )

            merged[f"{prefix}_centroid_progression_mps"] = (
                merged[f"{prefix}_centroid_x_smooth_m"].diff()
                / merged["delta_time_s"]
            )

        rows: list[dict] = []

        for row in merged.itertuples(index=False):
            possession_team = str(row.possession_team)

            attacking_team = (
                possession_team
                if possession_team in self.VALID_TEAMS
                else "unknown"
            )

            if attacking_team == "team_home":
                prefix = "home"
            elif attacking_team == "team_away":
                prefix = "away"
            else:
                prefix = None

            attack_direction = "unknown"
            progression_mps = np.nan
            attacking_width_m = np.nan
            attacking_length_m = np.nan
            attacking_compactness_m = np.nan
            attacking_visible_players = 0
            mean_speed_mps = np.nan
            max_speed_mps = np.nan
            overload_side = "unknown"
            final_third_entry = False
            transition_phase = "uncertain"
            confidence = 0.0

            if prefix is not None:
                progression_mps = getattr(
                    row,
                    f"{prefix}_centroid_progression_mps",
                )
                attacking_width_m = getattr(
                    row,
                    f"{prefix}_width_m",
                )
                attacking_length_m = getattr(
                    row,
                    f"{prefix}_length_m",
                )
                attacking_compactness_m = getattr(
                    row,
                    f"{prefix}_compactness_m",
                )
                attacking_visible_players = int(
                    getattr(
                        row,
                        f"{prefix}_visible_players",
                    )
                    if not pd.isna(
                        getattr(
                            row,
                            f"{prefix}_visible_players",
                        )
                    )
                    else 0
                )

                mean_speed_mps = getattr(
                    row,
                    f"{prefix}_mean_speed_from_tracks_mps",
                )
                max_speed_mps = getattr(
                    row,
                    f"{prefix}_max_speed_mps",
                )

                attack_direction = self._attack_direction(
                    progression_mps
                )

                overload_side = self._overload_side(
                    attacking_team=attacking_team,
                    frame_index=int(row.frame_index),
                    tracks=tracks,
                )

                final_third_entry = (
                    str(row.ball_zone) == "attacking_third"
                )

                transition_phase = self._classify_phase(
                    progression_mps=progression_mps,
                    mean_speed_mps=mean_speed_mps,
                    ball_zone=str(row.ball_zone),
                    possession_state=str(row.possession_state),
                    visible_players=attacking_visible_players,
                )

                confidence = self._confidence_score(
                    visible_players=attacking_visible_players,
                    possession_state=str(row.possession_state),
                    progression_mps=progression_mps,
                    mean_speed_mps=mean_speed_mps,
                )

            rows.append(
                {
                    "frame_index": int(row.frame_index),
                    "time_seconds": round(
                        float(row.time_seconds),
                        3,
                    ),
                    "attacking_team": attacking_team,
                    "possession_state": str(
                        row.possession_state
                    ),
                    "ball_x_m": (
                        round(float(row.ball_pitch_x_m), 3)
                        if not pd.isna(row.ball_pitch_x_m)
                        else np.nan
                    ),
                    "ball_y_m": (
                        round(float(row.ball_pitch_y_m), 3)
                        if not pd.isna(row.ball_pitch_y_m)
                        else np.nan
                    ),
                    "ball_zone": str(row.ball_zone),
                    "attack_direction": attack_direction,
                    "centroid_progression_mps": (
                        round(float(progression_mps), 3)
                        if not pd.isna(progression_mps)
                        else np.nan
                    ),
                    "attacking_width_m": (
                        round(float(attacking_width_m), 3)
                        if not pd.isna(attacking_width_m)
                        else np.nan
                    ),
                    "attacking_length_m": (
                        round(float(attacking_length_m), 3)
                        if not pd.isna(attacking_length_m)
                        else np.nan
                    ),
                    "attacking_compactness_m": (
                        round(
                            float(attacking_compactness_m),
                            3,
                        )
                        if not pd.isna(
                            attacking_compactness_m
                        )
                        else np.nan
                    ),
                    "attacking_visible_players":
                        attacking_visible_players,
                    "attacking_mean_speed_mps": (
                        round(float(mean_speed_mps), 3)
                        if not pd.isna(mean_speed_mps)
                        else np.nan
                    ),
                    "attacking_max_speed_mps": (
                        round(float(max_speed_mps), 3)
                        if not pd.isna(max_speed_mps)
                        else np.nan
                    ),
                    "overload_side": overload_side,
                    "final_third_entry": bool(
                        final_third_entry
                    ),
                    "transition_phase": transition_phase,
                    "analysis_confidence": round(
                        float(confidence),
                        3,
                    ),
                    "valid_transition_frame": bool(
                        attacking_team in self.VALID_TEAMS
                        and attacking_visible_players
                        >= self.minimum_visible_players
                    ),
                }
            )

        timeline = pd.DataFrame(rows)

        if not timeline.empty:
            timeline["phase_sequence_id"] = (
                timeline["transition_phase"]
                .ne(
                    timeline[
                        "transition_phase"
                    ].shift()
                )
                .cumsum()
            )

        return timeline

    def _build_summary(
        self,
        timeline: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one short-clip transition summary."""

        if timeline.empty:
            return pd.DataFrame()

        valid = timeline[
            timeline["valid_transition_frame"] == True
        ].copy()

        if valid.empty:
            valid = timeline.copy()

        dominant_team = self._mode_or_unknown(
            valid["attacking_team"]
        )
        dominant_phase = self._mode_or_unknown(
            valid["transition_phase"]
        )
        dominant_direction = self._mode_or_unknown(
            valid["attack_direction"]
        )
        dominant_overload = self._mode_or_unknown(
            valid["overload_side"]
        )

        phase_changes = int(
            valid["transition_phase"]
            .ne(
                valid[
                    "transition_phase"
                ].shift()
            )
            .sum()
            - 1
        )
        phase_changes = max(0, phase_changes)

        rows = [
            {
                "clip_type": "short_sequence",
                "frames_analysed": int(len(timeline)),
                "valid_transition_frames": int(
                    valid["valid_transition_frame"].sum()
                    if "valid_transition_frame"
                    in valid.columns
                    else len(valid)
                ),
                "dominant_attacking_team": dominant_team,
                "dominant_transition_phase": dominant_phase,
                "dominant_attack_direction": dominant_direction,
                "dominant_overload_side": dominant_overload,
                "mean_centroid_progression_mps":
                    self._mean(
                        valid[
                            "centroid_progression_mps"
                        ]
                    ),
                "mean_attacking_width_m":
                    self._mean(
                        valid[
                            "attacking_width_m"
                        ]
                    ),
                "mean_attacking_length_m":
                    self._mean(
                        valid[
                            "attacking_length_m"
                        ]
                    ),
                "mean_attacking_compactness_m":
                    self._mean(
                        valid[
                            "attacking_compactness_m"
                        ]
                    ),
                "mean_attacking_speed_mps":
                    self._mean(
                        valid[
                            "attacking_mean_speed_mps"
                        ]
                    ),
                "maximum_attacking_speed_mps":
                    self._max(
                        valid[
                            "attacking_max_speed_mps"
                        ]
                    ),
                "final_third_entry_frames": int(
                    valid[
                        "final_third_entry"
                    ].sum()
                ),
                "phase_changes": phase_changes,
                "mean_analysis_confidence":
                    self._mean(
                        valid[
                            "analysis_confidence"
                        ]
                    ),
                "interpretation": (
                    "Observed tactical transition only; "
                    "not a full-match formation claim."
                ),
            }
        ]

        return pd.DataFrame(rows)

    def _attack_direction(
        self,
        progression_mps: float,
    ) -> str:
        if pd.isna(progression_mps):
            return "unknown"

        if progression_mps >= self.progression_threshold_mps:
            return "positive_x"

        if progression_mps <= -self.progression_threshold_mps:
            return "negative_x"

        return "stable"

    def _classify_phase(
        self,
        progression_mps: float,
        mean_speed_mps: float,
        ball_zone: str,
        possession_state: str,
        visible_players: int,
    ) -> str:
        if visible_players < self.minimum_visible_players:
            return "insufficient_data"

        if possession_state in {"unknown", "pending"}:
            return "uncertain"

        if pd.isna(progression_mps):
            return "uncertain"

        speed = (
            float(mean_speed_mps)
            if not pd.isna(mean_speed_mps)
            else 0.0
        )

        if ball_zone == "attacking_third":
            if speed >= self.high_tempo_threshold_mps:
                return "final_third_attack"
            return "settled_final_third"

        if progression_mps >= self.progression_threshold_mps:
            if speed >= self.high_tempo_threshold_mps:
                return "fast_progression"
            return "controlled_progression"

        if progression_mps <= -self.progression_threshold_mps:
            return "retreat_or_recycle"

        return "stable_possession"

    def _overload_side(
        self,
        attacking_team: str,
        frame_index: int,
        tracks: pd.DataFrame,
    ) -> str:
        frame = tracks[
            (tracks["frame_index"] == frame_index)
            & (tracks["team_id"] == attacking_team)
        ]

        if frame.empty:
            return "unknown"

        median_y = float(
            frame["pitch_y_smooth_m"].median()
        )

        left_count = int(
            (
                frame["pitch_y_smooth_m"]
                < median_y
            ).sum()
        )
        right_count = int(
            (
                frame["pitch_y_smooth_m"]
                >= median_y
            ).sum()
        )

        if left_count > right_count:
            return "left"

        if right_count > left_count:
            return "right"

        return "balanced"

    def _confidence_score(
        self,
        visible_players: int,
        possession_state: str,
        progression_mps: float,
        mean_speed_mps: float,
    ) -> float:
        score = 0.0

        score += min(
            0.4,
            0.4
            * visible_players
            / max(self.minimum_visible_players, 1),
        )

        if possession_state in {
            "locked",
            "held",
            "switched",
        }:
            score += 0.3
        elif possession_state == "switching":
            score += 0.15

        if not pd.isna(progression_mps):
            score += 0.15

        if not pd.isna(mean_speed_mps):
            score += 0.15

        return min(1.0, score)

    def _build_report(
        self,
        tracks: pd.DataFrame,
        tactical: pd.DataFrame,
        possession: pd.DataFrame,
        timeline: pd.DataFrame,
        summary: pd.DataFrame,
        tracks_path: Path,
        tactical_path: Path,
        possession_path: Path,
        timeline_path: Path,
        summary_path: Path,
    ) -> dict:
        valid_frames = int(
            timeline["valid_transition_frame"].sum()
        ) if not timeline.empty else 0

        return {
            "analysis_type": "short_sequence_transition_analysis",
            "reference_input_video": "clean_match1.mp4",
            "tracks_input_file": str(tracks_path),
            "tactical_input_file": str(tactical_path),
            "possession_input_file": str(possession_path),
            "timeline_output_file": str(timeline_path),
            "summary_output_file": str(summary_path),
            "track_rows": int(len(tracks)),
            "tactical_rows": int(len(tactical)),
            "possession_rows": int(len(possession)),
            "timeline_rows": int(len(timeline)),
            "summary_rows": int(len(summary)),
            "valid_transition_frames": valid_frames,
            "invalid_or_uncertain_frames": int(
                len(timeline) - valid_frames
            ),
            "mean_analysis_confidence": (
                self._mean(
                    timeline["analysis_confidence"]
                )
                if not timeline.empty
                else None
            ),
            "important_limitation": (
                "The clip is short and does not support "
                "a full-match formation claim."
            ),
        }

    @staticmethod
    def _mode_or_unknown(series: pd.Series) -> str:
        values = (
            series.dropna()
            .astype(str)
        )

        values = values[
            ~values.isin(
                ["unknown", "uncertain"]
            )
        ]

        if values.empty:
            return "unknown"

        mode = values.mode()
        return (
            str(mode.iloc[0])
            if not mode.empty
            else "unknown"
        )

    @staticmethod
    def _mean(series: pd.Series) -> float:
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return round(
            float(values.mean()),
            3,
        )

    @staticmethod
    def _max(series: pd.Series) -> float:
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return np.nan

        return round(
            float(values.max()),
            3,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze short-sequence football transitions."
        )
    )

    parser.add_argument(
        "--tracks",
        default=(
            "outputs/analytics/"
            "cleaned_tracks.csv"
        ),
    )

    parser.add_argument(
        "--tactical",
        default=(
            "outputs/analytics/"
            "tactical_timeline.csv"
        ),
    )

    parser.add_argument(
        "--possession",
        default=(
            "outputs/analytics/"
            "possession_timeline_v2.csv"
        ),
    )

    parser.add_argument(
        "--timeline-output",
        default=(
            "outputs/analytics/"
            "transition_timeline.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "outputs/analytics/"
            "transition_summary.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "transition_report.json"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = TransitionAnalyzer()

    timeline, summary = analyzer.run(
        tracks_csv=args.tracks,
        tactical_csv=args.tactical,
        possession_csv=args.possession,
        timeline_output_csv=args.timeline_output,
        summary_output_csv=args.summary_output,
        report_output_json=args.report,
    )

    print("Transition analysis completed.")
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
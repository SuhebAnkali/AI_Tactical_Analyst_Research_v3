"""Player influence analysis for short football clips.

This module combines movement, possession, passing-network involvement,
transition participation, and spatial coverage into a normalized 0-100
influence score.

Designed for ``clean_match1.mp4`` and short broadcast sequences. Scores are
relative to the players visible in the analysed clip and must not be interpreted
as full-match ratings.

Inputs
------
- cleaned_tracks.csv
- player_statistics.csv
- possession_timeline_v2.csv
- transition_timeline.csv
- passing_network_nodes.csv

Outputs
-------
- player_influence.csv
- player_rankings.csv
- player_influence_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PLAYER_STATS_REQUIRED = {
    "track_id",
    "team_id",
    "active_frames",
    "total_distance_m",
    "average_speed_mps",
    "maximum_speed_mps",
    "sprint_count",
    "average_pitch_x_m",
    "average_pitch_y_m",
    "defensive_third_pct",
    "middle_third_pct",
    "attacking_third_pct",
    "mean_detection_confidence",
    "tracking_continuity_pct",
}

TRACKS_REQUIRED = {
    "frame_index",
    "time_seconds",
    "track_id",
    "class_name",
    "team_id",
    "pitch_x_smooth_m",
    "pitch_y_smooth_m",
}

POSSESSION_REQUIRED = {
    "frame_index",
    "time_seconds",
    "possession_team",
    "possession_owner_id",
    "possession_state",
}

TRANSITION_REQUIRED = {
    "frame_index",
    "attacking_team",
    "transition_phase",
    "valid_transition_frame",
    "analysis_confidence",
}

NETWORK_REQUIRED = {
    "team_id",
    "track_id",
    "passes_made",
    "passes_received",
    "out_degree",
    "in_degree",
    "total_network_involvement",
    "network_involvement_pct",
}


class PlayerInfluenceAnalyzer:
    """Calculate clip-relative player influence scores."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        movement_weight: float = 0.20,
        intensity_weight: float = 0.15,
        possession_weight: float = 0.20,
        passing_weight: float = 0.15,
        transition_weight: float = 0.15,
        spatial_weight: float = 0.15,
    ) -> None:
        weights = {
            "movement": movement_weight,
            "intensity": intensity_weight,
            "possession": possession_weight,
            "passing": passing_weight,
            "transition": transition_weight,
            "spatial": spatial_weight,
        }

        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError("Influence weights must sum to more than zero.")

        self.weights = {
            name: float(value) / total
            for name, value in weights.items()
        }

    def run(
        self,
        tracks_csv: str | Path,
        player_stats_csv: str | Path,
        possession_csv: str | Path,
        transition_csv: str | Path,
        network_nodes_csv: str | Path,
        influence_output_csv: str | Path,
        rankings_output_csv: str | Path,
        report_output_json: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        paths = {
            "tracks": Path(tracks_csv),
            "player statistics": Path(player_stats_csv),
            "possession": Path(possession_csv),
            "transition": Path(transition_csv),
            "network nodes": Path(network_nodes_csv),
        }

        for label, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")

        tracks = pd.read_csv(paths["tracks"])
        player_stats = pd.read_csv(paths["player statistics"])
        possession = pd.read_csv(paths["possession"])
        transition = pd.read_csv(paths["transition"])
        network = pd.read_csv(paths["network nodes"])

        self._validate(tracks, TRACKS_REQUIRED, "cleaned tracks")
        self._validate(
            player_stats,
            PLAYER_STATS_REQUIRED,
            "player statistics",
        )
        self._validate(
            possession,
            POSSESSION_REQUIRED,
            "possession timeline",
        )
        self._validate(
            transition,
            TRANSITION_REQUIRED,
            "transition timeline",
        )
        self._validate(
            network,
            NETWORK_REQUIRED,
            "passing-network nodes",
        )

        tracks = self._prepare_tracks(tracks)
        player_stats = self._prepare_player_stats(player_stats)
        possession = self._prepare_possession(possession)
        transition = self._prepare_transition(transition)
        network = self._prepare_network(network)

        influence = self._build_influence(
            tracks=tracks,
            player_stats=player_stats,
            possession=possession,
            transition=transition,
            network=network,
        )

        rankings = self._build_rankings(influence)

        influence_path = Path(influence_output_csv)
        rankings_path = Path(rankings_output_csv)
        report_path = Path(report_output_json)

        for path in (influence_path, rankings_path, report_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        influence.to_csv(influence_path, index=False)
        rankings.to_csv(rankings_path, index=False)

        report = self._build_report(
            influence=influence,
            rankings=rankings,
            paths=paths,
            influence_path=influence_path,
            rankings_path=rankings_path,
        )

        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return influence, rankings

    @staticmethod
    def _validate(
        data: pd.DataFrame,
        required: set[str],
        name: str,
    ) -> None:
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {name}: "
                + ", ".join(sorted(missing))
            )

    def _prepare_tracks(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        for column in (
            "frame_index",
            "time_seconds",
            "track_id",
            "pitch_x_smooth_m",
            "pitch_y_smooth_m",
        ):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["class_name"] = (
            df["class_name"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["team_id"] = (
            df["team_id"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            (df["class_name"] == "player")
            & df["team_id"].isin(self.VALID_TEAMS)
        ].copy()

        df = df.dropna(
            subset=[
                "frame_index",
                "track_id",
                "pitch_x_smooth_m",
                "pitch_y_smooth_m",
            ]
        )

        df["frame_index"] = df["frame_index"].astype(int)
        df["track_id"] = df["track_id"].astype(int)

        return df.sort_values(
            ["track_id", "frame_index"]
        ).reset_index(drop=True)

    def _prepare_player_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        numeric_columns = [
            column
            for column in PLAYER_STATS_REQUIRED
            if column not in {"team_id"}
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["team_id"] = (
            df["team_id"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[df["team_id"].isin(self.VALID_TEAMS)].copy()
        df["track_id"] = df["track_id"].astype(int)

        return df

    def _prepare_possession(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        for column in (
            "frame_index",
            "time_seconds",
            "possession_owner_id",
        ):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["possession_team"] = (
            df["possession_team"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["possession_state"] = (
            df["possession_state"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["delta_time_s"] = (
            df["time_seconds"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )

        return df

    def _prepare_transition(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        df["frame_index"] = pd.to_numeric(
            df["frame_index"],
            errors="coerce",
        )
        df["analysis_confidence"] = pd.to_numeric(
            df["analysis_confidence"],
            errors="coerce",
        ).fillna(0.0)

        df["attacking_team"] = (
            df["attacking_team"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        df["transition_phase"] = (
            df["transition_phase"]
            .fillna("uncertain")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["valid_transition_frame"] = (
            df["valid_transition_frame"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

        df = df.dropna(subset=["frame_index"])
        df["frame_index"] = df["frame_index"].astype(int)

        return df

    def _prepare_network(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        numeric_columns = [
            "track_id",
            "passes_made",
            "passes_received",
            "out_degree",
            "in_degree",
            "total_network_involvement",
            "network_involvement_pct",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            ).fillna(0.0)

        df["team_id"] = (
            df["team_id"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["track_id"] = df["track_id"].astype(int)

        return df[
            df["team_id"].isin(self.VALID_TEAMS)
        ].copy()

    def _build_influence(
        self,
        tracks: pd.DataFrame,
        player_stats: pd.DataFrame,
        possession: pd.DataFrame,
        transition: pd.DataFrame,
        network: pd.DataFrame,
    ) -> pd.DataFrame:
        spatial = self._spatial_metrics(tracks)
        possession_metrics = self._possession_metrics(possession)
        transition_metrics = self._transition_metrics(
            tracks,
            transition,
        )

        merged = (
            player_stats.merge(
                spatial,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                possession_metrics,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                transition_metrics,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                network[
                    [
                        "team_id",
                        "track_id",
                        "passes_made",
                        "passes_received",
                        "out_degree",
                        "in_degree",
                        "total_network_involvement",
                        "network_involvement_pct",
                    ]
                ],
                on=["team_id", "track_id"],
                how="left",
            )
        )

        fill_zero = [
            "coverage_area_m2",
            "zone_count",
            "possession_frames",
            "possession_time_sec",
            "estimated_touches",
            "transition_frames",
            "weighted_transition_participation",
            "passes_made",
            "passes_received",
            "out_degree",
            "in_degree",
            "total_network_involvement",
            "network_involvement_pct",
        ]

        for column in fill_zero:
            if column not in merged.columns:
                merged[column] = 0.0

        merged[fill_zero] = merged[fill_zero].fillna(0.0)

        merged["movement_raw"] = (
            0.70 * merged["total_distance_m"]
            + 0.30 * merged["active_frames"]
        )

        merged["intensity_raw"] = (
            0.45 * merged["average_speed_mps"]
            + 0.35 * merged["maximum_speed_mps"]
            + 0.20 * merged["sprint_count"]
        )

        merged["possession_raw"] = (
            0.65 * merged["possession_time_sec"]
            + 0.35 * merged["estimated_touches"]
        )

        merged["passing_raw"] = (
            0.35 * merged["passes_made"]
            + 0.25 * merged["passes_received"]
            + 0.20 * merged["out_degree"]
            + 0.20 * merged["in_degree"]
        )

        merged["transition_raw"] = (
            0.55 * merged["weighted_transition_participation"]
            + 0.45 * merged["transition_frames"]
        )

        merged["spatial_raw"] = (
            0.50 * merged["coverage_area_m2"]
            + 0.25 * merged["zone_count"]
            + 0.25 * merged["attacking_third_pct"]
        )

        component_map = {
            "movement_score": "movement_raw",
            "intensity_score": "intensity_raw",
            "possession_score": "possession_raw",
            "passing_score": "passing_raw",
            "transition_score": "transition_raw",
            "spatial_score": "spatial_raw",
        }

        for output_column, source_column in component_map.items():
            merged[output_column] = self._normalize_0_100(
                merged[source_column]
            )

        merged["influence_score"] = (
            self.weights["movement"] * merged["movement_score"]
            + self.weights["intensity"] * merged["intensity_score"]
            + self.weights["possession"] * merged["possession_score"]
            + self.weights["passing"] * merged["passing_score"]
            + self.weights["transition"] * merged["transition_score"]
            + self.weights["spatial"] * merged["spatial_score"]
        ).round(2)

        merged["data_quality_score"] = (
            0.45 * merged["tracking_continuity_pct"].clip(0, 100)
            + 0.35 * (100.0 * merged["mean_detection_confidence"]).clip(0, 100)
            + 0.20 * (100.0 * merged["team_assignment_ratio"]).clip(0, 100)
        ).round(2)

        merged["dominant_zone"] = merged.apply(
            self._dominant_zone,
            axis=1,
        )

        merged["left_right_bias"] = np.where(
            merged["average_pitch_y_m"] < 34.0,
            "left",
            "right",
        )

        merged["influence_band"] = pd.cut(
            merged["influence_score"],
            bins=[-np.inf, 25, 50, 75, np.inf],
            labels=["low", "moderate", "high", "very_high"],
        ).astype(str)

        keep_columns = [
            "track_id",
            "team_id",
            "influence_score",
            "influence_band",
            "data_quality_score",
            "movement_score",
            "intensity_score",
            "possession_score",
            "passing_score",
            "transition_score",
            "spatial_score",
            "total_distance_m",
            "average_speed_mps",
            "maximum_speed_mps",
            "sprint_count",
            "possession_frames",
            "possession_time_sec",
            "estimated_touches",
            "passes_made",
            "passes_received",
            "out_degree",
            "in_degree",
            "transition_frames",
            "weighted_transition_participation",
            "coverage_area_m2",
            "zone_count",
            "dominant_zone",
            "left_right_bias",
            "average_pitch_x_m",
            "average_pitch_y_m",
            "active_frames",
            "tracking_continuity_pct",
            "mean_detection_confidence",
        ]

        return merged[keep_columns].sort_values(
            ["influence_score", "data_quality_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

    def _spatial_metrics(
        self,
        tracks: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict] = []

        for (team_id, track_id), group in tracks.groupby(
            ["team_id", "track_id"]
        ):
            x = group["pitch_x_smooth_m"].dropna()
            y = group["pitch_y_smooth_m"].dropna()

            if x.empty or y.empty:
                coverage = 0.0
            else:
                coverage = float(
                    max(0.0, x.max() - x.min())
                    * max(0.0, y.max() - y.min())
                )

            zones = pd.cut(
                x,
                bins=[-np.inf, 35.0, 70.0, np.inf],
                labels=[
                    "defensive_third",
                    "middle_third",
                    "attacking_third",
                ],
            )

            rows.append(
                {
                    "team_id": team_id,
                    "track_id": int(track_id),
                    "coverage_area_m2": round(coverage, 3),
                    "zone_count": int(zones.nunique()),
                }
            )

        return pd.DataFrame(rows)

    def _possession_metrics(
        self,
        possession: pd.DataFrame,
    ) -> pd.DataFrame:
        owned = possession[
            possession["possession_team"].isin(self.VALID_TEAMS)
            & possession["possession_owner_id"].notna()
        ].copy()

        if owned.empty:
            return pd.DataFrame(
                columns=[
                    "team_id",
                    "track_id",
                    "possession_frames",
                    "possession_time_sec",
                    "estimated_touches",
                ]
            )

        owned["track_id"] = owned[
            "possession_owner_id"
        ].astype(int)
        owned["team_id"] = owned["possession_team"]

        owner_change = (
            owned["track_id"].ne(owned["track_id"].shift())
            | owned["team_id"].ne(owned["team_id"].shift())
        )
        owned["touch_start"] = owner_change.astype(int)

        return (
            owned.groupby(["team_id", "track_id"])
            .agg(
                possession_frames=("frame_index", "count"),
                possession_time_sec=("delta_time_s", "sum"),
                estimated_touches=("touch_start", "sum"),
            )
            .reset_index()
        )

    def _transition_metrics(
        self,
        tracks: pd.DataFrame,
        transition: pd.DataFrame,
    ) -> pd.DataFrame:
        valid = transition[
            transition["valid_transition_frame"]
            & transition["attacking_team"].isin(self.VALID_TEAMS)
        ][
            [
                "frame_index",
                "attacking_team",
                "analysis_confidence",
            ]
        ].copy()

        if valid.empty:
            return pd.DataFrame(
                columns=[
                    "team_id",
                    "track_id",
                    "transition_frames",
                    "weighted_transition_participation",
                ]
            )

        merged = tracks.merge(
            valid,
            on="frame_index",
            how="inner",
        )

        merged = merged[
            merged["team_id"] == merged["attacking_team"]
        ].copy()

        return (
            merged.groupby(["team_id", "track_id"])
            .agg(
                transition_frames=("frame_index", "nunique"),
                weighted_transition_participation=(
                    "analysis_confidence",
                    "sum",
                ),
            )
            .reset_index()
        )

    @staticmethod
    def _normalize_0_100(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).fillna(0.0)

        minimum = float(values.min())
        maximum = float(values.max())

        if np.isclose(maximum, minimum):
            return pd.Series(
                np.where(values > 0, 100.0, 0.0),
                index=values.index,
                dtype=float,
            )

        return 100.0 * (values - minimum) / (maximum - minimum)

    @staticmethod
    def _dominant_zone(row: pd.Series) -> str:
        values = {
            "defensive_third": float(
                row.get("defensive_third_pct", 0.0)
            ),
            "middle_third": float(
                row.get("middle_third_pct", 0.0)
            ),
            "attacking_third": float(
                row.get("attacking_third_pct", 0.0)
            ),
        }

        return max(values, key=values.get)

    @staticmethod
    def _build_rankings(
        influence: pd.DataFrame,
    ) -> pd.DataFrame:
        rankings = influence.copy()

        rankings["overall_rank"] = (
            rankings["influence_score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        rankings["team_rank"] = (
            rankings.groupby("team_id")["influence_score"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        columns = [
            "overall_rank",
            "team_rank",
            "track_id",
            "team_id",
            "influence_score",
            "influence_band",
            "data_quality_score",
            "movement_score",
            "intensity_score",
            "possession_score",
            "passing_score",
            "transition_score",
            "spatial_score",
            "dominant_zone",
            "left_right_bias",
        ]

        return rankings[columns].sort_values(
            ["overall_rank", "team_rank"]
        ).reset_index(drop=True)

    def _build_report(
        self,
        influence: pd.DataFrame,
        rankings: pd.DataFrame,
        paths: dict[str, Path],
        influence_path: Path,
        rankings_path: Path,
    ) -> dict:
        top_players = []

        for row in rankings.head(5).itertuples(index=False):
            top_players.append(
                {
                    "overall_rank": int(row.overall_rank),
                    "track_id": int(row.track_id),
                    "team_id": str(row.team_id),
                    "influence_score": float(row.influence_score),
                    "influence_band": str(row.influence_band),
                }
            )

        return {
            "analysis_type": "clip_relative_player_influence",
            "reference_input_video": "clean_match1.mp4",
            "tracks_input_file": str(paths["tracks"]),
            "player_statistics_input_file": str(
                paths["player statistics"]
            ),
            "possession_input_file": str(paths["possession"]),
            "transition_input_file": str(paths["transition"]),
            "network_nodes_input_file": str(paths["network nodes"]),
            "influence_output_file": str(influence_path),
            "rankings_output_file": str(rankings_path),
            "players_scored": int(len(influence)),
            "teams_present": sorted(
                influence["team_id"].dropna().unique().tolist()
            ),
            "weights": self.weights,
            "top_players": top_players,
            "important_limitation": (
                "Influence scores are relative to this short clip and "
                "must not be treated as full-match player ratings."
            ),
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate clip-relative football player influence scores."
        )
    )

    parser.add_argument(
        "--tracks",
        default="outputs/analytics/cleaned_tracks.csv",
    )
    parser.add_argument(
        "--player-stats",
        default="outputs/analytics/player_statistics.csv",
    )
    parser.add_argument(
        "--possession",
        default="outputs/analytics/possession_timeline_v2.csv",
    )
    parser.add_argument(
        "--transition",
        default="outputs/analytics/transition_timeline.csv",
    )
    parser.add_argument(
        "--network-nodes",
        default="outputs/analytics/passing_network_nodes.csv",
    )
    parser.add_argument(
        "--influence-output",
        default="outputs/analytics/player_influence.csv",
    )
    parser.add_argument(
        "--rankings-output",
        default="outputs/analytics/player_rankings.csv",
    )
    parser.add_argument(
        "--report",
        default="outputs/analytics/player_influence_report.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = PlayerInfluenceAnalyzer()

    influence, rankings = analyzer.run(
        tracks_csv=args.tracks,
        player_stats_csv=args.player_stats,
        possession_csv=args.possession,
        transition_csv=args.transition,
        network_nodes_csv=args.network_nodes,
        influence_output_csv=args.influence_output,
        rankings_output_csv=args.rankings_output,
        report_output_json=args.report,
    )

    print("Player influence analysis completed.")
    print("Players scored:", len(influence))
    print("Ranking rows:", len(rankings))
    print(
        "Influence output:",
        Path(args.influence_output).resolve(),
    )
    print(
        "Rankings output:",
        Path(args.rankings_output).resolve(),
    )
    print(
        "Report:",
        Path(args.report).resolve(),
    )


if __name__ == "__main__":
    main()

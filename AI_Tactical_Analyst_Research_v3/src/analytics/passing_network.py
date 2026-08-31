"""Estimate a passing network from short-sequence tracking analytics.

This module is designed for ``clean_match1.mp4`` and other short broadcast clips.
It infers *estimated passes* from possession-owner changes within the same team.

Inputs
------
- cleaned_tracks.csv
- possession_timeline_v2.csv

Outputs
-------
- pass_events.csv
- passing_network_edges.csv
- passing_network_nodes.csv
- passing_network_report.json

Important limitation
--------------------
These are estimated passes derived from tracking and possession transitions.
They are not official event annotations and should be reported as such.
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
}

POSSESSION_REQUIRED_COLUMNS = {
    "frame_index",
    "time_seconds",
    "possession_team",
    "possession_owner_id",
    "possession_state",
    "ball_pitch_x_m",
    "ball_pitch_y_m",
    "ball_zone",
}


class PassingNetworkAnalyzer:
    """Infer pass events and build player-level passing networks."""

    VALID_TEAMS = ("team_home", "team_away")

    def __init__(
        self,
        maximum_pass_gap_seconds: float = 1.5,
        minimum_owner_frames: int = 1,
    ) -> None:
        self.maximum_pass_gap_seconds = float(
            maximum_pass_gap_seconds
        )
        self.minimum_owner_frames = int(minimum_owner_frames)

    def run(
        self,
        tracks_csv: str | Path,
        possession_csv: str | Path,
        pass_events_output_csv: str | Path,
        edges_output_csv: str | Path,
        nodes_output_csv: str | Path,
        report_output_json: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Infer passes, aggregate network edges, and calculate node metrics."""

        tracks_path = Path(tracks_csv)
        possession_path = Path(possession_csv)

        if not tracks_path.exists():
            raise FileNotFoundError(
                f"Cleaned tracks file not found: {tracks_path}"
            )

        if not possession_path.exists():
            raise FileNotFoundError(
                f"Possession timeline file not found: {possession_path}"
            )

        tracks = pd.read_csv(tracks_path)
        possession = pd.read_csv(possession_path)

        self._validate_columns(
            tracks,
            TRACK_REQUIRED_COLUMNS,
            "cleaned tracks",
        )
        self._validate_columns(
            possession,
            POSSESSION_REQUIRED_COLUMNS,
            "possession timeline",
        )

        tracks = self._prepare_tracks(tracks)
        possession = self._prepare_possession(possession)

        pass_events = self._infer_pass_events(
            tracks=tracks,
            possession=possession,
        )
        edges = self._build_edges(pass_events)
        nodes = self._build_nodes(
            tracks=tracks,
            pass_events=pass_events,
            edges=edges,
        )

        pass_events_path = Path(pass_events_output_csv)
        edges_path = Path(edges_output_csv)
        nodes_path = Path(nodes_output_csv)
        report_path = Path(report_output_json)

        for path in (
            pass_events_path,
            edges_path,
            nodes_path,
            report_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)

        pass_events.to_csv(pass_events_path, index=False)
        edges.to_csv(edges_path, index=False)
        nodes.to_csv(nodes_path, index=False)

        report = self._build_report(
            tracks=tracks,
            possession=possession,
            pass_events=pass_events,
            edges=edges,
            nodes=nodes,
            tracks_path=tracks_path,
            possession_path=possession_path,
            pass_events_path=pass_events_path,
            edges_path=edges_path,
            nodes_path=nodes_path,
        )

        report_path.write_text(
            json.dumps(report, indent=4),
            encoding="utf-8",
        )

        return pass_events, edges, nodes

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

    def _prepare_tracks(
        self,
        tracks: pd.DataFrame,
    ) -> pd.DataFrame:
        data = tracks.copy()

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
            ["frame_index", "track_id"]
        ).reset_index(drop=True)

    def _prepare_possession(
        self,
        possession: pd.DataFrame,
    ) -> pd.DataFrame:
        data = possession.copy()

        numeric_columns = [
            "frame_index",
            "time_seconds",
            "possession_owner_id",
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

        data = data.dropna(
            subset=[
                "frame_index",
                "time_seconds",
            ]
        )

        data["frame_index"] = data["frame_index"].astype(int)

        return data.sort_values(
            ["frame_index", "time_seconds"]
        ).reset_index(drop=True)

    def _infer_pass_events(
        self,
        tracks: pd.DataFrame,
        possession: pd.DataFrame,
    ) -> pd.DataFrame:
        """Infer a pass when possession changes between teammates."""

        valid = possession[
            possession["possession_team"].isin(self.VALID_TEAMS)
            & possession["possession_owner_id"].notna()
        ].copy()

        if valid.empty:
            return self._empty_pass_events()

        valid["possession_owner_id"] = (
            valid["possession_owner_id"].astype(int)
        )

        # Compress repeated ownership into owner segments.
        segment_start = (
            valid["possession_team"].ne(
                valid["possession_team"].shift()
            )
            | valid["possession_owner_id"].ne(
                valid["possession_owner_id"].shift()
            )
        )

        valid["owner_segment_id"] = segment_start.cumsum()

        segments = (
            valid.groupby("owner_segment_id")
            .agg(
                team_id=("possession_team", "first"),
                owner_id=("possession_owner_id", "first"),
                start_frame=("frame_index", "min"),
                end_frame=("frame_index", "max"),
                start_time_sec=("time_seconds", "min"),
                end_time_sec=("time_seconds", "max"),
                owner_frames=("frame_index", "count"),
                start_ball_x_m=("ball_pitch_x_m", "first"),
                start_ball_y_m=("ball_pitch_y_m", "first"),
                end_ball_x_m=("ball_pitch_x_m", "last"),
                end_ball_y_m=("ball_pitch_y_m", "last"),
                start_zone=("ball_zone", "first"),
                end_zone=("ball_zone", "last"),
            )
            .reset_index(drop=True)
        )

        rows: list[dict] = []
        event_id = 1

        for index in range(len(segments) - 1):
            current = segments.iloc[index]
            following = segments.iloc[index + 1]

            if int(current["owner_frames"]) < self.minimum_owner_frames:
                continue

            if int(following["owner_frames"]) < self.minimum_owner_frames:
                continue

            if str(current["team_id"]) != str(following["team_id"]):
                continue

            passer_id = int(current["owner_id"])
            receiver_id = int(following["owner_id"])

            if passer_id == receiver_id:
                continue

            gap_seconds = float(
                following["start_time_sec"]
                - current["end_time_sec"]
            )

            if gap_seconds < 0:
                continue

            if gap_seconds > self.maximum_pass_gap_seconds:
                continue

            start_x = self._safe_float(current["end_ball_x_m"])
            start_y = self._safe_float(current["end_ball_y_m"])
            end_x = self._safe_float(following["start_ball_x_m"])
            end_y = self._safe_float(following["start_ball_y_m"])

            pass_distance = self._distance(
                start_x,
                start_y,
                end_x,
                end_y,
            )

            pass_direction = self._pass_direction(
                start_x,
                end_x,
            )

            passer_position = self._player_position_near_frame(
                tracks=tracks,
                track_id=passer_id,
                frame_index=int(current["end_frame"]),
            )
            receiver_position = self._player_position_near_frame(
                tracks=tracks,
                track_id=receiver_id,
                frame_index=int(following["start_frame"]),
            )

            rows.append(
                {
                    "pass_event_id": event_id,
                    "team_id": str(current["team_id"]),
                    "passer_id": passer_id,
                    "receiver_id": receiver_id,
                    "start_frame": int(current["end_frame"]),
                    "end_frame": int(following["start_frame"]),
                    "start_time_sec": round(
                        float(current["end_time_sec"]),
                        3,
                    ),
                    "end_time_sec": round(
                        float(following["start_time_sec"]),
                        3,
                    ),
                    "gap_time_sec": round(gap_seconds, 3),
                    "pass_start_x_m": start_x,
                    "pass_start_y_m": start_y,
                    "pass_end_x_m": end_x,
                    "pass_end_y_m": end_y,
                    "pass_distance_m": pass_distance,
                    "pass_direction": pass_direction,
                    "start_zone": str(current["end_zone"]),
                    "end_zone": str(following["start_zone"]),
                    "passer_pitch_x_m": passer_position[0],
                    "passer_pitch_y_m": passer_position[1],
                    "receiver_pitch_x_m": receiver_position[0],
                    "receiver_pitch_y_m": receiver_position[1],
                    "estimated_pass": True,
                }
            )

            event_id += 1

        if not rows:
            return self._empty_pass_events()

        return pd.DataFrame(rows)

    @staticmethod
    def _empty_pass_events() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "pass_event_id",
                "team_id",
                "passer_id",
                "receiver_id",
                "start_frame",
                "end_frame",
                "start_time_sec",
                "end_time_sec",
                "gap_time_sec",
                "pass_start_x_m",
                "pass_start_y_m",
                "pass_end_x_m",
                "pass_end_y_m",
                "pass_distance_m",
                "pass_direction",
                "start_zone",
                "end_zone",
                "passer_pitch_x_m",
                "passer_pitch_y_m",
                "receiver_pitch_x_m",
                "receiver_pitch_y_m",
                "estimated_pass",
            ]
        )

    def _build_edges(
        self,
        pass_events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate passes into directed player-to-player edges."""

        if pass_events.empty:
            return pd.DataFrame(
                columns=[
                    "team_id",
                    "passer_id",
                    "receiver_id",
                    "pass_count",
                    "mean_pass_distance_m",
                    "total_pass_distance_m",
                    "first_pass_time_sec",
                    "last_pass_time_sec",
                ]
            )

        edges = (
            pass_events.groupby(
                ["team_id", "passer_id", "receiver_id"]
            )
            .agg(
                pass_count=("pass_event_id", "count"),
                mean_pass_distance_m=(
                    "pass_distance_m",
                    "mean",
                ),
                total_pass_distance_m=(
                    "pass_distance_m",
                    "sum",
                ),
                first_pass_time_sec=(
                    "start_time_sec",
                    "min",
                ),
                last_pass_time_sec=(
                    "end_time_sec",
                    "max",
                ),
            )
            .reset_index()
        )

        for column in (
            "mean_pass_distance_m",
            "total_pass_distance_m",
            "first_pass_time_sec",
            "last_pass_time_sec",
        ):
            edges[column] = pd.to_numeric(
                edges[column],
                errors="coerce",
            ).round(3)

        return edges.sort_values(
            ["team_id", "pass_count"],
            ascending=[True, False],
        ).reset_index(drop=True)

    def _build_nodes(
        self,
        tracks: pd.DataFrame,
        pass_events: pd.DataFrame,
        edges: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate simple directed-network metrics for each player."""

        base = (
            tracks.groupby(["team_id", "track_id"])
            .agg(
                average_pitch_x_m=(
                    "pitch_x_smooth_m",
                    "mean",
                ),
                average_pitch_y_m=(
                    "pitch_y_smooth_m",
                    "mean",
                ),
                visible_frames=("frame_index", "count"),
            )
            .reset_index()
        )

        if pass_events.empty:
            base["passes_made"] = 0
            base["passes_received"] = 0
            base["out_degree"] = 0
            base["in_degree"] = 0
            base["total_network_involvement"] = 0
            base["network_involvement_pct"] = 0.0
            return base.sort_values(
                ["team_id", "track_id"]
            ).reset_index(drop=True)

        passes_made = (
            pass_events.groupby(
                ["team_id", "passer_id"]
            )
            .size()
            .rename("passes_made")
            .reset_index()
            .rename(columns={"passer_id": "track_id"})
        )

        passes_received = (
            pass_events.groupby(
                ["team_id", "receiver_id"]
            )
            .size()
            .rename("passes_received")
            .reset_index()
            .rename(columns={"receiver_id": "track_id"})
        )

        out_degree = (
            edges.groupby(
                ["team_id", "passer_id"]
            )["receiver_id"]
            .nunique()
            .rename("out_degree")
            .reset_index()
            .rename(columns={"passer_id": "track_id"})
        )

        in_degree = (
            edges.groupby(
                ["team_id", "receiver_id"]
            )["passer_id"]
            .nunique()
            .rename("in_degree")
            .reset_index()
            .rename(columns={"receiver_id": "track_id"})
        )

        nodes = (
            base.merge(
                passes_made,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                passes_received,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                out_degree,
                on=["team_id", "track_id"],
                how="left",
            )
            .merge(
                in_degree,
                on=["team_id", "track_id"],
                how="left",
            )
        )

        fill_columns = [
            "passes_made",
            "passes_received",
            "out_degree",
            "in_degree",
        ]

        nodes[fill_columns] = (
            nodes[fill_columns]
            .fillna(0)
            .astype(int)
        )

        nodes["total_network_involvement"] = (
            nodes["passes_made"]
            + nodes["passes_received"]
        )

        team_total_involvement = (
            nodes.groupby("team_id")[
                "total_network_involvement"
            ]
            .transform("sum")
        )

        nodes["network_involvement_pct"] = np.where(
            team_total_involvement > 0,
            100.0
            * nodes["total_network_involvement"]
            / team_total_involvement,
            0.0,
        )

        numeric_round_columns = [
            "average_pitch_x_m",
            "average_pitch_y_m",
            "network_involvement_pct",
        ]

        nodes[numeric_round_columns] = (
            nodes[numeric_round_columns].round(3)
        )

        return nodes.sort_values(
            [
                "team_id",
                "total_network_involvement",
                "track_id",
            ],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    def _build_report(
        self,
        tracks: pd.DataFrame,
        possession: pd.DataFrame,
        pass_events: pd.DataFrame,
        edges: pd.DataFrame,
        nodes: pd.DataFrame,
        tracks_path: Path,
        possession_path: Path,
        pass_events_path: Path,
        edges_path: Path,
        nodes_path: Path,
    ) -> dict:
        team_pass_counts = (
            pass_events["team_id"].value_counts().to_dict()
            if not pass_events.empty
            else {}
        )

        return {
            "analysis_type": "estimated_short_sequence_passing_network",
            "reference_input_video": "clean_match1.mp4",
            "tracks_input_file": str(tracks_path),
            "possession_input_file": str(possession_path),
            "pass_events_output_file": str(pass_events_path),
            "edges_output_file": str(edges_path),
            "nodes_output_file": str(nodes_path),
            "track_rows": int(len(tracks)),
            "possession_rows": int(len(possession)),
            "estimated_pass_events": int(len(pass_events)),
            "network_edges": int(len(edges)),
            "network_nodes": int(len(nodes)),
            "team_pass_counts": {
                str(key): int(value)
                for key, value in team_pass_counts.items()
            },
            "maximum_pass_gap_seconds":
                self.maximum_pass_gap_seconds,
            "minimum_owner_frames":
                self.minimum_owner_frames,
            "important_limitation": (
                "Passes are estimated from possession-owner changes "
                "within a short clip and are not official event labels."
            ),
        }

    def _player_position_near_frame(
        self,
        tracks: pd.DataFrame,
        track_id: int,
        frame_index: int,
    ) -> tuple[float, float]:
        player = tracks[
            tracks["track_id"] == track_id
        ].copy()

        if player.empty:
            return np.nan, np.nan

        player["frame_difference"] = (
            player["frame_index"] - frame_index
        ).abs()

        nearest = player.sort_values(
            "frame_difference"
        ).iloc[0]

        return (
            round(float(nearest["pitch_x_smooth_m"]), 3),
            round(float(nearest["pitch_y_smooth_m"]), 3),
        )

    @staticmethod
    def _distance(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        values = [x1, y1, x2, y2]

        if any(pd.isna(value) for value in values):
            return np.nan

        return round(
            float(np.hypot(x2 - x1, y2 - y1)),
            3,
        )

    @staticmethod
    def _pass_direction(
        start_x: float,
        end_x: float,
    ) -> str:
        if pd.isna(start_x) or pd.isna(end_x):
            return "unknown"

        difference = float(end_x - start_x)

        if difference > 2.0:
            return "forward"

        if difference < -2.0:
            return "backward"

        return "lateral"

    @staticmethod
    def _safe_float(value: object) -> float:
        if pd.isna(value):
            return np.nan

        return round(float(value), 3)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate passes and build a short-sequence "
            "football passing network."
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
        "--possession",
        default=(
            "outputs/analytics/"
            "possession_timeline_v2.csv"
        ),
    )

    parser.add_argument(
        "--pass-events-output",
        default=(
            "outputs/analytics/"
            "pass_events.csv"
        ),
    )

    parser.add_argument(
        "--edges-output",
        default=(
            "outputs/analytics/"
            "passing_network_edges.csv"
        ),
    )

    parser.add_argument(
        "--nodes-output",
        default=(
            "outputs/analytics/"
            "passing_network_nodes.csv"
        ),
    )

    parser.add_argument(
        "--report",
        default=(
            "outputs/analytics/"
            "passing_network_report.json"
        ),
    )

    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--minimum-owner-frames",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    analyzer = PassingNetworkAnalyzer(
        maximum_pass_gap_seconds=args.max_gap_seconds,
        minimum_owner_frames=args.minimum_owner_frames,
    )

    pass_events, edges, nodes = analyzer.run(
        tracks_csv=args.tracks,
        possession_csv=args.possession,
        pass_events_output_csv=args.pass_events_output,
        edges_output_csv=args.edges_output,
        nodes_output_csv=args.nodes_output,
        report_output_json=args.report,
    )

    print("Passing network analysis completed.")
    print("Estimated pass events:", len(pass_events))
    print("Network edges:", len(edges))
    print("Network nodes:", len(nodes))
    print(
        "Pass events output:",
        Path(args.pass_events_output).resolve(),
    )
    print(
        "Edges output:",
        Path(args.edges_output).resolve(),
    )
    print(
        "Nodes output:",
        Path(args.nodes_output).resolve(),
    )
    print(
        "Report:",
        Path(args.report).resolve(),
    )


if __name__ == "__main__":
    main()

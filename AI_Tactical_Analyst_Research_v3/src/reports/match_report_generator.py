"""Generate an expanded 12-20 page football analytics research report.

This module replaces the earlier compact match report generator. It combines the
project's CSV outputs into a research-style PDF containing methodology, system
architecture, pipeline diagram, dataset description, preprocessing, tracking,
analytics, conclusions, future work, and multiple visualizations.

Reference input
---------------
clean_match1.mp4

Primary outputs
---------------
- outputs/reports/match_report_detailed.pdf
- outputs/reports/match_summary_detailed.json

Visualizations
--------------
- Player heatmaps
- Team heatmaps
- Ball trajectory
- Passing network graph
- Tactical timeline
- Transition timeline
- Speed distribution
- Distance covered
- Influence distribution
- Estimated possession comparison
- Pipeline architecture diagram

Important
---------
The source video is a short football clip. All possession, passing, transition,
and influence statistics are estimates derived from computer-vision tracking.
They must not be presented as official full-match statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class DetailedMatchReportGenerator:
    """Create a research-style football analytics PDF and JSON summary."""

    def __init__(
        self,
        report_title: str = "AI Football Tactical Analytics Research Report",
        reference_video: str = "clean_match1.mp4",
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        self.report_title = report_title
        self.reference_video = reference_video
        self.pitch_length_m = float(pitch_length_m)
        self.pitch_width_m = float(pitch_width_m)

    def run(
        self,
        cleaned_tracks_csv: str | Path,
        ball_csv: str | Path,
        player_statistics_csv: str | Path,
        team_statistics_csv: str | Path,
        tactical_timeline_csv: str | Path,
        tactical_summary_csv: str | Path,
        possession_timeline_csv: str | Path,
        possession_summary_csv: str | Path,
        transition_timeline_csv: str | Path,
        transition_summary_csv: str | Path,
        passing_edges_csv: str | Path,
        passing_nodes_csv: str | Path,
        player_influence_csv: str | Path,
        player_rankings_csv: str | Path,
        preprocessing_report_json: str | Path,
        feature_report_json: str | Path,
        tactical_report_json: str | Path,
        possession_report_json: str | Path,
        transition_report_json: str | Path,
        passing_report_json: str | Path,
        influence_report_json: str | Path,
        pdf_output: str | Path,
        json_output: str | Path,
    ) -> tuple[Path, Path]:
        paths = {
            "cleaned_tracks": Path(cleaned_tracks_csv),
            "ball": Path(ball_csv),
            "player_statistics": Path(player_statistics_csv),
            "team_statistics": Path(team_statistics_csv),
            "tactical_timeline": Path(tactical_timeline_csv),
            "tactical_summary": Path(tactical_summary_csv),
            "possession_timeline": Path(possession_timeline_csv),
            "possession_summary": Path(possession_summary_csv),
            "transition_timeline": Path(transition_timeline_csv),
            "transition_summary": Path(transition_summary_csv),
            "passing_edges": Path(passing_edges_csv),
            "passing_nodes": Path(passing_nodes_csv),
            "player_influence": Path(player_influence_csv),
            "player_rankings": Path(player_rankings_csv),
            "preprocessing_report": Path(preprocessing_report_json),
            "feature_report": Path(feature_report_json),
            "tactical_report": Path(tactical_report_json),
            "possession_report": Path(possession_report_json),
            "transition_report": Path(transition_report_json),
            "passing_report": Path(passing_report_json),
            "influence_report": Path(influence_report_json),
        }

        required = [
            "cleaned_tracks",
            "ball",
            "player_statistics",
            "team_statistics",
            "tactical_timeline",
            "tactical_summary",
            "possession_timeline",
            "possession_summary",
            "transition_timeline",
            "transition_summary",
            "passing_edges",
            "passing_nodes",
            "player_influence",
            "player_rankings",
        ]

        for key in required:
            if not paths[key].exists():
                raise FileNotFoundError(
                    f"Required input file not found ({key}): {paths[key]}"
                )

        csv_data = {
            key: pd.read_csv(path)
            for key, path in paths.items()
            if path.suffix.lower() == ".csv" and path.exists()
        }

        json_data = {
            key: self._read_json(path)
            for key, path in paths.items()
            if path.suffix.lower() == ".json" and path.exists()
        }

        pdf_path = Path(pdf_output)
        summary_path = Path(json_output)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self._build_summary(csv_data, json_data)
        summary_path.write_text(
            json.dumps(summary, indent=4),
            encoding="utf-8",
        )

        self._build_pdf(
            csv_data=csv_data,
            json_data=json_data,
            summary=summary,
            pdf_path=pdf_path,
        )

        return pdf_path, summary_path

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _build_summary(
        self,
        data: dict[str, pd.DataFrame],
        reports: dict[str, dict],
    ) -> dict:
        player_stats = data["player_statistics"]
        team_stats = data["team_statistics"]
        possession = data["possession_summary"]
        transition = data["transition_summary"]
        rankings = data["player_rankings"]
        passing_edges = data["passing_edges"]

        top_players = []
        for row in rankings.head(10).itertuples(index=False):
            top_players.append(
                {
                    "overall_rank": int(getattr(row, "overall_rank", 0)),
                    "track_id": int(getattr(row, "track_id", 0)),
                    "team_id": str(getattr(row, "team_id", "unknown")),
                    "influence_score": float(
                        getattr(row, "influence_score", 0.0)
                    ),
                }
            )

        possession_summary = {}
        for row in possession.itertuples(index=False):
            team_id = str(getattr(row, "team_id", "unknown"))
            possession_summary[team_id] = {
                "possession_frames": int(
                    getattr(row, "possession_frames", 0)
                ),
                "possession_time_sec": float(
                    getattr(row, "possession_time_sec", 0.0)
                ),
                "possession_pct_known": float(
                    getattr(row, "possession_pct_known", 0.0)
                ),
            }

        transition_summary = {}
        if not transition.empty:
            row = transition.iloc[0]
            transition_summary = {
                "dominant_attacking_team": str(
                    row.get("dominant_attacking_team", "unknown")
                ),
                "dominant_transition_phase": str(
                    row.get("dominant_transition_phase", "unknown")
                ),
                "dominant_attack_direction": str(
                    row.get("dominant_attack_direction", "unknown")
                ),
                "dominant_overload_side": str(
                    row.get("dominant_overload_side", "unknown")
                ),
                "mean_analysis_confidence": float(
                    row.get("mean_analysis_confidence", 0.0)
                ),
            }

        return {
            "report_title": self.report_title,
            "reference_video": self.reference_video,
            "analysis_scope": "short_sequence_clip_analysis",
            "frame_level_rows": int(
                len(data.get("cleaned_tracks", pd.DataFrame()))
            ),
            "players_in_player_statistics": int(len(player_stats)),
            "teams_present": sorted(
                team_stats["team_id"].astype(str).unique().tolist()
            )
            if not team_stats.empty and "team_id" in team_stats
            else [],
            "estimated_passing_edges": int(len(passing_edges)),
            "top_players": top_players,
            "possession_summary": possession_summary,
            "transition_summary": transition_summary,
            "source_reports": reports,
            "limitations": [
                "The reference input is a short football clip, not a full match.",
                "Possession and passes are estimated from tracking data.",
                "Observed transition analysis replaces full formation claims.",
                "Pitch-space metrics depend on homography accuracy.",
                "Player influence values are relative to this clip.",
                "Manual ground-truth annotation is required for formal validation.",
            ],
        }

    def _build_pdf(
        self,
        csv_data: dict[str, pd.DataFrame],
        json_data: dict[str, dict],
        summary: dict,
        pdf_path: Path,
    ) -> None:
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=21,
            leading=25,
            spaceAfter=14,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Heading2"],
            alignment=TA_CENTER,
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#44546A"),
            spaceAfter=14,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=6,
            spaceAfter=8,
        )
        subheading_style = ParagraphStyle(
            "Subheading",
            parent=styles["Heading3"],
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#1F4E78"),
            spaceBefore=5,
            spaceAfter=5,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=13.5,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
        note_style = ParagraphStyle(
            "Note",
            parent=body_style,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#666666"),
        )

        document = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=1.35 * cm,
            leftMargin=1.35 * cm,
            topMargin=1.25 * cm,
            bottomMargin=1.25 * cm,
            title=self.report_title,
            author="AI Tactical Analyst Research Project",
        )

        story: list = []

        # Cover page
        story.append(Spacer(1, 2.0 * cm))
        story.append(Paragraph(self.report_title, title_style))
        story.append(
            Paragraph(
                "Computer Vision, Tracking, Data Preprocessing and Tactical Analytics",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 0.8 * cm))
        cover_rows = [
            ["Reference video", self.reference_video],
            ["Analysis scope", "Short-sequence football clip"],
            ["Primary method", "YOLO detection + ByteTrack + analytics"],
            ["Report type", "Research-oriented technical report"],
        ]
        story.append(
            self._styled_table(
                cover_rows,
                header=False,
                widths=[5.2 * cm, 10.8 * cm],
            )
        )
        story.append(Spacer(1, 1.0 * cm))
        story.append(
            Paragraph(
                "<b>Important:</b> This document presents estimated analytics "
                "derived from computer-vision tracking. It does not claim official "
                "full-match statistics.",
                body_style,
            )
        )
        story.append(PageBreak())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            figures = self._create_all_figures(csv_data, temp)

            # Executive summary
            story.append(Paragraph("1. Executive Summary", heading_style))
            story.append(
                Paragraph(
                    "This project implements an end-to-end AI football tactical "
                    "analytics pipeline. The system detects players, referees and "
                    "the ball, maintains object identities over time, assigns teams "
                    "using shirt-colour features, converts frame-level tracking into "
                    "structured datasets, and generates tactical, possession, "
                    "transition, passing-network and player-influence analytics.",
                    body_style,
                )
            )
            story.append(
                self._styled_table(
                    [
                        ["Metric", "Result"],
                        [
                            "Cleaned tracking rows",
                            str(len(csv_data["cleaned_tracks"])),
                        ],
                        [
                            "Player statistics rows",
                            str(len(csv_data["player_statistics"])),
                        ],
                        [
                            "Tactical frames",
                            str(len(csv_data["tactical_timeline"])),
                        ],
                        [
                            "Possession frames with ball",
                            str(len(csv_data["possession_timeline"])),
                        ],
                        [
                            "Estimated passing edges",
                            str(len(csv_data["passing_edges"])),
                        ],
                        [
                            "Players scored for influence",
                            str(len(csv_data["player_influence"])),
                        ],
                    ],
                    widths=[8.0 * cm, 8.0 * cm],
                )
            )
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "The report is deliberately framed as short-clip analysis. "
                    "Formation claims are avoided because the input does not show "
                    "all outfield players consistently for a settled tactical phase.",
                    note_style,
                )
            )

            # Architecture and methodology
            story.append(PageBreak())
            story.append(Paragraph("2. System Architecture", heading_style))
            story.append(
                Paragraph(
                    "The architecture follows a layered design. The computer-vision "
                    "layer produces detections and tracks; the data layer cleans and "
                    "transforms records; the analytics layer creates football metrics; "
                    "the reporting layer produces tables, charts and downloadable outputs.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("pipeline"), 17.0, 10.0)

            story.append(Paragraph("3. Methodology", heading_style))
            methodology = [
                ["Stage", "Method", "Purpose"],
                [
                    "Detection",
                    "YOLO football model",
                    "Detect players, referee and ball",
                ],
                [
                    "Tracking",
                    "ByteTrack-inspired association",
                    "Maintain stable track IDs",
                ],
                [
                    "Team assignment",
                    "Shirt-colour embeddings",
                    "Assign home and away labels",
                ],
                [
                    "Pitch mapping",
                    "Homography",
                    "Estimate metric pitch coordinates",
                ],
                [
                    "Preprocessing",
                    "Validation, interpolation, smoothing",
                    "Create a reliable analytics dataset",
                ],
                [
                    "Analytics",
                    "Spatial, temporal and network features",
                    "Generate tactical insights",
                ],
            ]
            story.append(self._styled_table(methodology))

            # Dataset
            story.append(PageBreak())
            story.append(Paragraph("4. Dataset and Input Video", heading_style))
            story.append(
                Paragraph(
                    f"The reference input is <b>{self.reference_video}</b>. "
                    "The current development outputs are based on the validated "
                    "test segment used during module development. Before final "
                    "submission, the same pipeline should be rerun on the complete clip.",
                    body_style,
                )
            )
            dataset_rows = self._dataset_overview_rows(csv_data, json_data)
            story.append(self._styled_table(dataset_rows))
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "The generated tracking CSV is treated as a derived research "
                    "dataset. Each row represents an object observation at a specific "
                    "frame and contains identity, team, bounding-box, pitch-position, "
                    "speed and confidence information.",
                    body_style,
                )
            )

            # Preprocessing
            story.append(PageBreak())
            story.append(Paragraph("5. Data Preprocessing", heading_style))
            story.append(
                Paragraph(
                    "The preprocessing stage validates schema and data types, removes "
                    "invalid duplicate observations, checks coordinate ranges, rejects "
                    "impossible speeds, interpolates short pitch-coordinate gaps, applies "
                    "trajectory smoothing and recalculates movement features.",
                    body_style,
                )
            )
            preprocessing = json_data.get("preprocessing_report", {})
            story.append(
                self._json_key_value_table(
                    preprocessing,
                    preferred=[
                        "original_rows",
                        "cleaned_rows",
                        "removed_rows",
                        "original_unique_tracks",
                        "cleaned_unique_tracks",
                        "unknown_team_rows",
                        "missing_pitch_coordinate_rows",
                    ],
                )
            )
            story.append(Paragraph("Preprocessing Output Fields", subheading_style))
            story.append(
                self._styled_table(
                    [
                        ["Feature", "Description"],
                        ["pitch_x_smooth_m", "Smoothed horizontal pitch coordinate"],
                        ["pitch_y_smooth_m", "Smoothed vertical pitch coordinate"],
                        ["step_distance_m", "Estimated displacement between observations"],
                        ["calculated_speed_mps", "Smoothed estimated player speed"],
                        ["pitch_zone", "Defensive, middle or attacking third"],
                        ["is_sprint", "Threshold-based sprint indicator"],
                    ]
                )
            )

            # Detection and tracking
            story.append(PageBreak())
            story.append(Paragraph("6. Detection and Tracking", heading_style))
            story.append(
                Paragraph(
                    "The detector identifies football entities. The tracker uses "
                    "high-confidence association, lower-confidence recovery, lost-track "
                    "reactivation, class-aware matching, motion prediction and separate "
                    "ball tracking. Stable IDs are essential because team labels and "
                    "analytics accumulate across multiple frames.",
                    body_style,
                )
            )
            story.append(
                self._dataframe_table(
                    csv_data["team_statistics"],
                    [
                        "team_id",
                        "tracked_players",
                        "total_active_frames",
                        "total_distance_m",
                        "average_speed_mps",
                        "maximum_speed_mps",
                        "total_sprints",
                    ],
                )
            )
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "Unknown team labels are preserved rather than force-assigned. "
                    "This avoids fabricating data when shirt colour cannot be extracted "
                    "reliably because of motion blur, occlusion or small player size.",
                    note_style,
                )
            )

            # Player and team heatmaps
            story.append(PageBreak())
            story.append(Paragraph("7. Spatial Analytics and Heatmaps", heading_style))
            story.append(
                Paragraph(
                    "Heatmaps summarize where players and teams were observed on the "
                    "estimated pitch. Density is calculated from smoothed pitch coordinates.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("team_heatmaps"), 17.0, 8.0)
            story.append(Spacer(1, 8))
            self._append_image(story, figures.get("player_heatmaps"), 17.0, 11.0)

            # Ball trajectory and possession
            story.append(PageBreak())
            story.append(Paragraph("8. Ball Tracking and Possession", heading_style))
            story.append(
                Paragraph(
                    "Ball ownership is estimated in image space using the distance "
                    "between the ball centre and player foot positions, normalized by "
                    "player bounding-box height. Temporal smoothing prevents rapid "
                    "one-frame possession changes. Pitch coordinates are used for "
                    "field-zone analysis rather than primary ownership assignment.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("ball_trajectory"), 17.0, 8.2)
            story.append(Spacer(1, 8))
            self._append_image(story, figures.get("possession"), 16.0, 7.0)
            story.append(
                self._dataframe_table(
                    csv_data["possession_summary"],
                    [
                        "team_id",
                        "possession_frames",
                        "possession_time_sec",
                        "possession_pct_known",
                        "possession_sequences",
                        "longest_sequence_sec",
                    ],
                )
            )

            # Tactical timeline
            story.append(PageBreak())
            story.append(Paragraph("9. Tactical Frame Analytics", heading_style))
            story.append(
                Paragraph(
                    "Frame-level tactical metrics include team centroid, width, length, "
                    "compactness, visible-player count, attacking line, defensive line, "
                    "pitch occupation and distance between team centroids.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("tactical_timeline"), 17.0, 10.0)
            story.append(
                self._dataframe_table(
                    csv_data["tactical_summary"],
                    [
                        "team_id",
                        "mean_visible_players",
                        "mean_width_m",
                        "mean_length_m",
                        "mean_compactness_m",
                        "mean_pitch_occupation_pct",
                    ],
                )
            )

            # Transition timeline
            story.append(PageBreak())
            story.append(Paragraph("10. Transition Analysis", heading_style))
            story.append(
                Paragraph(
                    "Because the input is a short attacking clip, the system reports "
                    "observed tactical transition phases rather than a fixed formation. "
                    "Metrics include centroid progression, attacking speed, width, "
                    "compactness, overload side and final-third presence.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("transition_timeline"), 17.0, 9.5)
            story.append(
                self._dataframe_table(
                    csv_data["transition_summary"],
                    [
                        "dominant_attacking_team",
                        "dominant_transition_phase",
                        "dominant_attack_direction",
                        "dominant_overload_side",
                        "mean_attacking_speed_mps",
                        "mean_analysis_confidence",
                    ],
                )
            )

            # Passing network
            story.append(PageBreak())
            story.append(Paragraph("11. Estimated Passing Network", heading_style))
            story.append(
                Paragraph(
                    "A pass is inferred when possession ownership changes between "
                    "different players of the same team within a short time gap. "
                    "This is an estimated network, not official event annotation.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("passing_network"), 16.5, 9.5)
            story.append(
                self._dataframe_table(
                    csv_data["passing_edges"],
                    [
                        "team_id",
                        "passer_id",
                        "receiver_id",
                        "pass_count",
                        "mean_pass_distance_m",
                        "total_pass_distance_m",
                    ],
                )
            )

            # Player performance charts
            story.append(PageBreak())
            story.append(Paragraph("12. Player Movement and Performance", heading_style))
            self._append_image(story, figures.get("speed_distribution"), 16.5, 7.5)
            story.append(Spacer(1, 8))
            self._append_image(story, figures.get("distance_covered"), 16.5, 8.0)

            # Influence
            story.append(PageBreak())
            story.append(Paragraph("13. Player Influence Analysis", heading_style))
            story.append(
                Paragraph(
                    "The influence score combines movement, intensity, estimated "
                    "possession, passing involvement, transition participation and "
                    "spatial coverage. Scores are normalized relative to players "
                    "visible in this clip.",
                    body_style,
                )
            )
            self._append_image(story, figures.get("influence_distribution"), 16.5, 8.0)
            story.append(
                self._dataframe_table(
                    csv_data["player_rankings"].head(12),
                    [
                        "overall_rank",
                        "team_rank",
                        "track_id",
                        "team_id",
                        "influence_score",
                        "influence_band",
                        "data_quality_score",
                    ],
                    max_rows=12,
                )
            )

            # Results and discussion
            story.append(PageBreak())
            story.append(Paragraph("14. Results and Discussion", heading_style))
            story.append(
                Paragraph(
                    "The system successfully produced structured outputs for tracking, "
                    "team assignment, ball trajectory, tactical frame metrics, estimated "
                    "possession, transition phases, passing-network relationships and "
                    "player influence. The strongest methodological improvement was the "
                    "use of image-space normalized distance for possession estimation, "
                    "which reduced unknown possession compared with a fixed pitch-distance "
                    "threshold.",
                    body_style,
                )
            )
            story.append(Paragraph("Key Findings", subheading_style))
            findings = self._key_findings(csv_data)
            story.append(
                self._styled_table(
                    [["Finding", "Observed result"]] + findings,
                    widths=[7.5 * cm, 8.5 * cm],
                )
            )
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "The analytics are internally consistent for demonstration and "
                    "prototype evaluation. Formal research validation would require "
                    "manual ground truth for detection, identity tracking, team labels, "
                    "ball ownership and pass events.",
                    body_style,
                )
            )

            # Limitations
            story.append(PageBreak())
            story.append(Paragraph("15. Limitations", heading_style))
            for limitation in summary["limitations"]:
                story.append(Paragraph(f"- {limitation}", body_style))
            story.append(Spacer(1, 8))
            story.append(
                Paragraph(
                    "The most important limitation is the short duration and changing "
                    "broadcast camera context. A short attacking sequence cannot support "
                    "stable full-match formation, workload or possession conclusions.",
                    body_style,
                )
            )

            # Conclusion and future work
            story.append(Paragraph("16. Conclusion", heading_style))
            story.append(
                Paragraph(
                    "The project demonstrates a complete pipeline that transforms a "
                    "football video into structured tactical analytics. It goes beyond "
                    "object detection by integrating identity tracking, preprocessing, "
                    "feature engineering, spatial analysis, possession, transition, "
                    "passing-network and player-influence modules. The resulting system "
                    "is suitable as a final-year research prototype and portfolio project.",
                    body_style,
                )
            )
            story.append(Paragraph("17. Future Work", heading_style))
            future_rows = [
                ["Priority", "Future improvement"],
                ["High", "Run and validate the complete input clip"],
                ["High", "Add Streamlit upload and progress workflow"],
                ["High", "Evaluate detection and tracking against manual ground truth"],
                ["Medium", "Improve homography and camera-motion compensation"],
                ["Medium", "Train a stronger football-specific ball detector"],
                ["Medium", "Support multiple matches and match history"],
                ["Advanced", "Add Voronoi control and pressure estimation"],
                ["Advanced", "Add simplified expected-threat pitch zones"],
                ["Advanced", "Compare rule-based analytics with learned models"],
            ]
            story.append(self._styled_table(future_rows))

            story.append(Spacer(1, 10))
            story.append(
                Paragraph(
                    "Final interpretation: all values in this report are experimental "
                    "estimates from the current short clip. They are appropriate for "
                    "demonstrating the system architecture and analytics workflow, but "
                    "must be validated before professional sporting use.",
                    note_style,
                )
            )

            document.build(story)

    def _create_all_figures(
        self,
        data: dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            "pipeline": output_dir / "pipeline.png",
            "team_heatmaps": output_dir / "team_heatmaps.png",
            "player_heatmaps": output_dir / "player_heatmaps.png",
            "ball_trajectory": output_dir / "ball_trajectory.png",
            "possession": output_dir / "possession.png",
            "passing_network": output_dir / "passing_network.png",
            "tactical_timeline": output_dir / "tactical_timeline.png",
            "transition_timeline": output_dir / "transition_timeline.png",
            "speed_distribution": output_dir / "speed_distribution.png",
            "distance_covered": output_dir / "distance_covered.png",
            "influence_distribution": output_dir / "influence_distribution.png",
        }

        self._plot_pipeline(figures["pipeline"])
        self._plot_team_heatmaps(data["cleaned_tracks"], figures["team_heatmaps"])
        self._plot_player_heatmaps(
            data["cleaned_tracks"],
            data["player_rankings"],
            figures["player_heatmaps"],
        )
        self._plot_ball_trajectory(data["ball"], figures["ball_trajectory"])
        self._plot_possession(data["possession_summary"], figures["possession"])
        self._plot_passing_network(
            data["passing_edges"],
            data["passing_nodes"],
            figures["passing_network"],
        )
        self._plot_tactical_timeline(
            data["tactical_timeline"],
            figures["tactical_timeline"],
        )
        self._plot_transition_timeline(
            data["transition_timeline"],
            figures["transition_timeline"],
        )
        self._plot_speed_distribution(
            data["player_statistics"],
            figures["speed_distribution"],
        )
        self._plot_distance_covered(
            data["player_statistics"],
            figures["distance_covered"],
        )
        self._plot_influence_distribution(
            data["player_influence"],
            figures["influence_distribution"],
        )

        return figures

    @staticmethod
    def _plot_pipeline(path: Path) -> None:
        stages = [
            "Video Input",
            "YOLO Detection",
            "ByteTrack",
            "Team + Ball",
            "Preprocessing",
            "Feature Engineering",
            "Tactical Analytics",
            "Report + Dashboard",
        ]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis("off")

        xs = np.linspace(0.06, 0.94, len(stages))
        y = 0.5
        for index, (x, label) in enumerate(zip(xs, stages)):
            box = FancyBboxPatch(
                (x - 0.055, y - 0.12),
                0.11,
                0.24,
                boxstyle="round,pad=0.015",
                linewidth=1.2,
                facecolor="#D9EAF7",
                edgecolor="#1F4E78",
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                wrap=True,
            )
            if index < len(stages) - 1:
                arrow = FancyArrowPatch(
                    (x + 0.057, y),
                    (xs[index + 1] - 0.057, y),
                    arrowstyle="->",
                    mutation_scale=12,
                    linewidth=1.2,
                )
                ax.add_patch(arrow)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("End-to-End AI Football Analytics Pipeline", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)

    def _plot_pitch(self, ax) -> None:
        ax.set_xlim(0, self.pitch_length_m)
        ax.set_ylim(0, self.pitch_width_m)
        ax.set_aspect("equal")
        ax.plot(
            [0, self.pitch_length_m, self.pitch_length_m, 0, 0],
            [0, 0, self.pitch_width_m, self.pitch_width_m, 0],
            linewidth=1,
        )
        ax.axvline(self.pitch_length_m / 2, linewidth=0.8)
        centre = plt.Circle(
            (self.pitch_length_m / 2, self.pitch_width_m / 2),
            9.15,
            fill=False,
            linewidth=0.8,
        )
        ax.add_patch(centre)
        ax.set_xlabel("Pitch length (m)")
        ax.set_ylabel("Pitch width (m)")

    def _plot_team_heatmaps(self, tracks: pd.DataFrame, path: Path) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, team in zip(axes, ["team_home", "team_away"]):
            self._plot_pitch(ax)
            subset = tracks[
                (tracks.get("class_name", "") == "player")
                & (tracks.get("team_id", "") == team)
            ]
            if not subset.empty:
                ax.hexbin(
                    subset["pitch_x_smooth_m"],
                    subset["pitch_y_smooth_m"],
                    gridsize=20,
                    mincnt=1,
                )
            ax.set_title(f"{team} spatial density")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def _plot_player_heatmaps(
        self,
        tracks: pd.DataFrame,
        rankings: pd.DataFrame,
        path: Path,
    ) -> None:
        if rankings.empty:
            return
        top_ids = rankings.head(4)["track_id"].astype(int).tolist()
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
        for ax, track_id in zip(axes.flatten(), top_ids):
            self._plot_pitch(ax)
            subset = tracks[
                pd.to_numeric(tracks["track_id"], errors="coerce") == track_id
            ]
            if not subset.empty:
                ax.hexbin(
                    subset["pitch_x_smooth_m"],
                    subset["pitch_y_smooth_m"],
                    gridsize=14,
                    mincnt=1,
                )
                ax.plot(
                    subset["pitch_x_smooth_m"],
                    subset["pitch_y_smooth_m"],
                    linewidth=0.8,
                    alpha=0.7,
                )
            ax.set_title(f"Player track {track_id}")
        fig.suptitle("Top-Ranked Player Heatmaps")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def _plot_ball_trajectory(self, ball: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        self._plot_pitch(ax)
        if not ball.empty:
            x_col = "pitch_x_m" if "pitch_x_m" in ball else "ball_pitch_x_m"
            y_col = "pitch_y_m" if "pitch_y_m" in ball else "ball_pitch_y_m"
            if x_col in ball and y_col in ball:
                ordered = ball.sort_values("frame_index")
                ax.plot(
                    ordered[x_col],
                    ordered[y_col],
                    marker="o",
                    markersize=2.5,
                    linewidth=1.0,
                )
                if len(ordered) > 0:
                    ax.scatter(
                        ordered[x_col].iloc[0],
                        ordered[y_col].iloc[0],
                        marker="s",
                        s=45,
                        label="Start",
                    )
                    ax.scatter(
                        ordered[x_col].iloc[-1],
                        ordered[y_col].iloc[-1],
                        marker="X",
                        s=55,
                        label="End",
                    )
                    ax.legend()
        ax.set_title("Estimated Ball Trajectory")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_possession(possession: pd.DataFrame, path: Path) -> None:
        if possession.empty or "possession_pct_known" not in possession:
            return
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(
            possession["team_id"].astype(str),
            possession["possession_pct_known"],
        )
        ax.set_ylabel("Estimated possession (%)")
        ax.set_title("Estimated Possession - Known Frames")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def _plot_passing_network(
        self,
        edges: pd.DataFrame,
        nodes: pd.DataFrame,
        path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        self._plot_pitch(ax)

        if not nodes.empty:
            x_name = (
                "average_pitch_x_m"
                if "average_pitch_x_m" in nodes
                else None
            )
            y_name = (
                "average_pitch_y_m"
                if "average_pitch_y_m" in nodes
                else None
            )
            if x_name and y_name:
                involved_ids: set[int] = set()
                if not edges.empty:
                    involved_ids.update(edges["passer_id"].astype(int).tolist())
                    involved_ids.update(edges["receiver_id"].astype(int).tolist())

                display_nodes = nodes.copy()
                if involved_ids:
                    display_nodes = display_nodes[
                        display_nodes["track_id"].astype(int).isin(involved_ids)
                    ]

                if not display_nodes.empty:
                    sizes = (
                        pd.to_numeric(
                            display_nodes.get(
                                "total_network_involvement",
                                1,
                            ),
                            errors="coerce",
                        )
                        .fillna(0)
                        .to_numpy()
                    )
                    sizes = 80 + 120 * sizes
                    ax.scatter(
                        display_nodes[x_name],
                        display_nodes[y_name],
                        s=sizes,
                        alpha=0.8,
                    )
                    for row in display_nodes.itertuples(index=False):
                        ax.text(
                            float(getattr(row, x_name)),
                            float(getattr(row, y_name)),
                            str(int(getattr(row, "track_id"))),
                            ha="center",
                            va="center",
                            fontsize=8,
                        )

                    if not edges.empty:
                        node_lookup = {
                            int(row.track_id): (
                                float(getattr(row, x_name)),
                                float(getattr(row, y_name)),
                            )
                            for row in display_nodes.itertuples(index=False)
                        }
                        for edge in edges.itertuples(index=False):
                            p = int(getattr(edge, "passer_id"))
                            r = int(getattr(edge, "receiver_id"))
                            if p in node_lookup and r in node_lookup:
                                x1, y1 = node_lookup[p]
                                x2, y2 = node_lookup[r]
                                ax.annotate(
                                    "",
                                    xy=(x2, y2),
                                    xytext=(x1, y1),
                                    arrowprops={
                                        "arrowstyle": "->",
                                        "linewidth": 1.2,
                                    },
                                )

        ax.set_title("Estimated Passing Network")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_tactical_timeline(timeline: pd.DataFrame, path: Path) -> None:
        if timeline.empty:
            return
        fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
        time = timeline["time_seconds"]

        for team, prefix in [("Home", "home"), ("Away", "away")]:
            if f"{prefix}_width_m" in timeline:
                axes[0].plot(
                    time,
                    timeline[f"{prefix}_width_m"],
                    label=team,
                )
            if f"{prefix}_length_m" in timeline:
                axes[1].plot(
                    time,
                    timeline[f"{prefix}_length_m"],
                    label=team,
                )
            if f"{prefix}_compactness_m" in timeline:
                axes[2].plot(
                    time,
                    timeline[f"{prefix}_compactness_m"],
                    label=team,
                )

        axes[0].set_ylabel("Width (m)")
        axes[1].set_ylabel("Length (m)")
        axes[2].set_ylabel("Compactness (m)")
        axes[2].set_xlabel("Time (s)")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.25)
        fig.suptitle("Tactical Shape Timeline")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_transition_timeline(
        transition: pd.DataFrame,
        path: Path,
    ) -> None:
        if transition.empty:
            return
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
        time = transition["time_seconds"]

        if "centroid_progression_mps" in transition:
            axes[0].plot(time, transition["centroid_progression_mps"])
            axes[0].axhline(0, linewidth=0.8)
            axes[0].set_ylabel("Progression (m/s)")

        if "analysis_confidence" in transition:
            axes[1].plot(time, transition["analysis_confidence"])
            axes[1].set_ylim(0, 1.05)
            axes[1].set_ylabel("Confidence")
            axes[1].set_xlabel("Time (s)")

        for ax in axes:
            ax.grid(alpha=0.25)
        fig.suptitle("Observed Transition Timeline")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_speed_distribution(player_stats: pd.DataFrame, path: Path) -> None:
        if player_stats.empty or "average_speed_mps" not in player_stats:
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for team in player_stats["team_id"].dropna().unique():
            subset = player_stats[player_stats["team_id"] == team]
            ax.hist(
                subset["average_speed_mps"].dropna(),
                bins=8,
                alpha=0.55,
                label=str(team),
            )
        ax.set_xlabel("Average speed (m/s)")
        ax.set_ylabel("Player count")
        ax.set_title("Player Speed Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_distance_covered(player_stats: pd.DataFrame, path: Path) -> None:
        if player_stats.empty or "total_distance_m" not in player_stats:
            return
        chart = player_stats.nlargest(12, "total_distance_m").copy()
        chart = chart.sort_values("total_distance_m")
        labels = (
            chart["team_id"].astype(str)
            + " - ID "
            + chart["track_id"].astype(int).astype(str)
        )
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.barh(labels, chart["total_distance_m"])
        ax.set_xlabel("Estimated distance (m)")
        ax.set_title("Top Players by Distance Covered")
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    @staticmethod
    def _plot_influence_distribution(
        influence: pd.DataFrame,
        path: Path,
    ) -> None:
        if influence.empty or "influence_score" not in influence:
            return
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        axes[0].hist(influence["influence_score"].dropna(), bins=8)
        axes[0].set_xlabel("Influence score")
        axes[0].set_ylabel("Player count")
        axes[0].set_title("Influence Score Distribution")

        top = influence.nlargest(10, "influence_score").sort_values(
            "influence_score"
        )
        labels = "ID " + top["track_id"].astype(int).astype(str)
        axes[1].barh(labels, top["influence_score"])
        axes[1].set_xlabel("Influence score")
        axes[1].set_title("Top Player Influence")

        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)

    def _dataset_overview_rows(
        self,
        data: dict[str, pd.DataFrame],
        reports: dict[str, dict],
    ) -> list[list[str]]:
        tracks = data["cleaned_tracks"]
        ball = data["ball"]
        rows = [
            ["Dataset component", "Rows", "Description"],
            [
                "Cleaned player/referee tracks",
                str(len(tracks)),
                "Frame-level object tracking dataset",
            ],
            [
                "Ball tracks",
                str(len(ball)),
                "Frame-level ball observations",
            ],
            [
                "Player statistics",
                str(len(data["player_statistics"])),
                "One row per retained player track",
            ],
            [
                "Tactical timeline",
                str(len(data["tactical_timeline"])),
                "One row per analysed frame",
            ],
            [
                "Possession timeline",
                str(len(data["possession_timeline"])),
                "Frames containing tracked ball data",
            ],
        ]
        return rows

    def _key_findings(
        self,
        data: dict[str, pd.DataFrame],
    ) -> list[list[str]]:
        findings: list[list[str]] = []

        possession = data["possession_summary"]
        if not possession.empty and "possession_pct_known" in possession:
            best = possession.loc[
                possession["possession_pct_known"].idxmax()
            ]
            findings.append(
                [
                    "Estimated possession leader",
                    f"{best['team_id']} ({best['possession_pct_known']:.1f}%)",
                ]
            )

        transition = data["transition_summary"]
        if not transition.empty:
            row = transition.iloc[0]
            findings.append(
                [
                    "Dominant transition",
                    str(row.get("dominant_transition_phase", "unknown")),
                ]
            )

        rankings = data["player_rankings"]
        if not rankings.empty:
            row = rankings.iloc[0]
            findings.append(
                [
                    "Highest influence",
                    (
                        f"Track {int(row['track_id'])} - "
                        f"{float(row['influence_score']):.2f}"
                    ),
                ]
            )

        team_stats = data["team_statistics"]
        if not team_stats.empty and "total_distance_m" in team_stats:
            best = team_stats.loc[team_stats["total_distance_m"].idxmax()]
            findings.append(
                [
                    "Greater team distance",
                    f"{best['team_id']} ({best['total_distance_m']:.2f} m)",
                ]
            )

        findings.append(
            [
                "Estimated pass relationships",
                str(len(data["passing_edges"])),
            ]
        )
        return findings

    def _json_key_value_table(
        self,
        data: dict,
        preferred: Iterable[str],
    ) -> Table:
        rows = [["Metric", "Value"]]
        for key in preferred:
            if key in data:
                rows.append([key, str(data[key])])
        if len(rows) == 1:
            rows.append(["Status", "No report values available"])
        return self._styled_table(rows, widths=[8 * cm, 8 * cm])

    def _dataframe_table(
        self,
        dataframe: pd.DataFrame,
        preferred_columns: list[str],
        max_rows: int = 12,
    ) -> Table:
        if dataframe.empty:
            return self._styled_table(
                [["Status"], ["No data available"]]
            )

        columns = [
            column
            for column in preferred_columns
            if column in dataframe.columns
        ]
        if not columns:
            columns = dataframe.columns[:6].tolist()

        view = dataframe[columns].head(max_rows).copy()

        for column in view.columns:
            if pd.api.types.is_float_dtype(view[column]):
                view[column] = view[column].round(3)

        rows = [columns] + view.fillna("").astype(str).values.tolist()
        total_width = 17.0 * cm
        width = total_width / max(len(columns), 1)

        return self._styled_table(
            rows,
            widths=[width] * len(columns),
        )

    @staticmethod
    def _styled_table(
        rows: list[list[str]],
        header: bool = True,
        widths: list[float] | None = None,
    ) -> Table:
        table = Table(
            rows,
            colWidths=widths,
            repeatRows=1 if header else 0,
        )
        commands = [
            ("GRID", (0, 0), (-1, -1), 0.45, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            (
                "ROWBACKGROUNDS",
                (0, 1 if header else 0),
                (-1, -1),
                [colors.white, colors.HexColor("#EAF2F8")],
            ),
        ]
        if header:
            commands.extend(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#1F4E78"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        table.setStyle(TableStyle(commands))
        return table

    @staticmethod
    def _append_image(
        story: list,
        path: Path | None,
        width_cm: float,
        height_cm: float,
    ) -> None:
        if path is not None and path.exists():
            story.append(
                Image(
                    str(path),
                    width=width_cm * cm,
                    height=height_cm * cm,
                )
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an expanded research-style football analytics PDF."
        )
    )

    parser.add_argument(
        "--cleaned-tracks",
        default="outputs/analytics/cleaned_tracks.csv",
    )
    parser.add_argument(
        "--ball",
        default="outputs/tracker_v2/ball.csv",
    )
    parser.add_argument(
        "--player-statistics",
        default="outputs/analytics/player_statistics.csv",
    )
    parser.add_argument(
        "--team-statistics",
        default="outputs/analytics/team_statistics.csv",
    )
    parser.add_argument(
        "--tactical-timeline",
        default="outputs/analytics/tactical_timeline.csv",
    )
    parser.add_argument(
        "--tactical-summary",
        default="outputs/analytics/tactical_summary.csv",
    )
    parser.add_argument(
        "--possession-timeline",
        default="outputs/analytics/possession_timeline_v2.csv",
    )
    parser.add_argument(
        "--possession-summary",
        default="outputs/analytics/possession_summary_v2.csv",
    )
    parser.add_argument(
        "--transition-timeline",
        default="outputs/analytics/transition_timeline.csv",
    )
    parser.add_argument(
        "--transition-summary",
        default="outputs/analytics/transition_summary.csv",
    )
    parser.add_argument(
        "--passing-edges",
        default="outputs/analytics/passing_network_edges.csv",
    )
    parser.add_argument(
        "--passing-nodes",
        default="outputs/analytics/passing_network_nodes.csv",
    )
    parser.add_argument(
        "--player-influence",
        default="outputs/analytics/player_influence.csv",
    )
    parser.add_argument(
        "--player-rankings",
        default="outputs/analytics/player_rankings.csv",
    )

    parser.add_argument(
        "--preprocessing-report",
        default="outputs/analytics/preprocessing_report.json",
    )
    parser.add_argument(
        "--feature-report",
        default="outputs/analytics/feature_engineering_report.json",
    )
    parser.add_argument(
        "--tactical-report",
        default="outputs/analytics/tactical_analytics_report.json",
    )
    parser.add_argument(
        "--possession-report",
        default="outputs/analytics/possession_report_v2.json",
    )
    parser.add_argument(
        "--transition-report",
        default="outputs/analytics/transition_report.json",
    )
    parser.add_argument(
        "--passing-report",
        default="outputs/analytics/passing_network_report.json",
    )
    parser.add_argument(
        "--influence-report",
        default="outputs/analytics/player_influence_report.json",
    )

    parser.add_argument(
        "--pdf-output",
        default="outputs/reports/match_report_detailed.pdf",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/reports/match_summary_detailed.json",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    generator = DetailedMatchReportGenerator()

    pdf_path, json_path = generator.run(
        cleaned_tracks_csv=args.cleaned_tracks,
        ball_csv=args.ball,
        player_statistics_csv=args.player_statistics,
        team_statistics_csv=args.team_statistics,
        tactical_timeline_csv=args.tactical_timeline,
        tactical_summary_csv=args.tactical_summary,
        possession_timeline_csv=args.possession_timeline,
        possession_summary_csv=args.possession_summary,
        transition_timeline_csv=args.transition_timeline,
        transition_summary_csv=args.transition_summary,
        passing_edges_csv=args.passing_edges,
        passing_nodes_csv=args.passing_nodes,
        player_influence_csv=args.player_influence,
        player_rankings_csv=args.player_rankings,
        preprocessing_report_json=args.preprocessing_report,
        feature_report_json=args.feature_report,
        tactical_report_json=args.tactical_report,
        possession_report_json=args.possession_report,
        transition_report_json=args.transition_report,
        passing_report_json=args.passing_report,
        influence_report_json=args.influence_report,
        pdf_output=args.pdf_output,
        json_output=args.json_output,
    )

    print("Detailed match report generated.")
    print("PDF:", pdf_path.resolve())
    print("JSON:", json_path.resolve())


if __name__ == "__main__":
    main()
"""Report generation utilities, including Power BI guide PDF."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.core.constants import POWERBI_DIR, REPORT_DIR


class ReportGenerator:
    """Generate Power BI guide and summary reports."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def generate_powerbi_guide(self, path: str | Path | None = None) -> Path:
        destination = Path(path) if path else POWERBI_DIR / "PowerBI_Dashboard_Guide.pdf"
        destination.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(destination), pagesize=A4, title="Power BI Dashboard Guide")
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="BodySmall", fontSize=9, leading=13))
        story = [Paragraph("AI Tactical Analyst Research v3 - Power BI Dashboard Guide", styles["Title"]), Spacer(1, 12)]
        sections = [
            ("Import", "Use Get Data > Text/CSV to load tracks.csv, player_stats.csv, ball.csv, events.csv, formations.csv, timeline.csv, and tracking.csv. Keep UTF-8 encoding and retain numeric columns as decimal numbers."),
            ("Relationships", "Create relationships using frame_index between timeline/events/formations/ball/tracking. Use track_id between player_stats and tracks/tracking. Use team_id between formations and tracking/team aggregates."),
            ("Measures", "Recommended DAX measures: TotalDistance = SUM(player_stats[distance_m]); AvgSpeed = AVERAGE(player_stats[mean_speed_mps]); PossessionPct = DIVIDE(SUM(player_stats[possession_seconds]), SUM(player_stats[possession_seconds]) + CALCULATE(SUM(player_stats[possession_seconds]), ALL(player_stats[team_id])))."),
            ("KPIs", "Build cards for possession, total passes, shots, goal attempts, sprint count, average distance per player, and event counts per phase. Use slicers for team_id, track_id, and half / time bucket."),
            ("Charts", "Recommended visuals: event timeline line chart, formation over time matrix, distance by player bar chart, scatter plot of average speed vs distance, heatmap image viewer, and cumulative possession area chart."),
            ("DAX", "Example: Shots = CALCULATE(COUNTROWS(events), events[event_type] = \"shot\"); GoalAttempts = CALCULATE(COUNTROWS(events), events[event_type] = \"goal_attempt\"); HomePossession = CALCULATE(SUM(player_stats[possession_seconds]), player_stats[team_id] IN {\"team_home\",\"goalkeeper_home\"})."),
        ]
        for title, body in sections:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Paragraph(body, styles["BodySmall"]))
            story.append(Spacer(1, 10))
        table_data = [["CSV", "Primary Purpose"], ["tracks.csv", "One row per active track per frame"], ["tracking.csv", "Expanded tactical state with pitch coordinates"], ["player_stats.csv", "Aggregated player metrics"], ["events.csv", "Event-level heuristic detections"], ["formations.csv", "Formation timeline per team"], ["ball.csv", "Ball trajectory and velocity"], ["timeline.csv", "Frame-level summary"]]
        table = Table(table_data, colWidths=[120, 320])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fafc")),
        ]))
        story.append(table)
        doc.build(story)
        return destination

    def write_summary_markdown(self, outputs: dict[str, Path], stats_frame: pd.DataFrame) -> Path:
        destination = REPORT_DIR / "summary_report.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# AI Tactical Analyst Summary Report", "", "## Outputs"]
        for name, path in outputs.items():
            lines.append(f"- **{name}**: `{path}`")
        lines += ["", "## Top Distance Leaders"]
        if not stats_frame.empty and "distance_m" in stats_frame.columns:
            for _, row in stats_frame.sort_values("distance_m", ascending=False).head(10).iterrows():
                lines.append(f"- Track {row['track_id']} ({row['team_id']}): {row['distance_m']:.2f} m")
        destination.write_text("\n".join(lines), encoding="utf-8")
        return destination

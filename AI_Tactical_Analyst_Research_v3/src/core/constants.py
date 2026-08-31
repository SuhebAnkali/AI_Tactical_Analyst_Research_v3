"""Shared constants for the tactical analysis system."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "AI_Tactical_Analyst_Research_v3"
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "configs"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
REPORT_DIR = OUTPUT_DIR / "reports"
POWERBI_DIR = ROOT_DIR / "powerbi"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_AREA_LENGTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.3
GOAL_AREA_LENGTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
CENTER_CIRCLE_RADIUS_M = 9.15

CLASS_NAMES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
}

TEAM_LABELS = ["team_home", "team_away", "goalkeeper_home", "goalkeeper_away", "referee", "unknown"]

FORMATION_TEMPLATES = {
    "4-4-2": [4, 4, 2],
    "4-3-3": [4, 3, 3],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "3-4-3": [3, 4, 3],
    "4-1-4-1": [4, 1, 4, 1],
    "4-5-1": [4, 5, 1],
    "5-3-2": [5, 3, 2],
    "5-4-1": [5, 4, 1],
}

CSV_EXPORTS = [
    "tracks.csv",
    "formations.csv",
    "player_stats.csv",
    "ball.csv",
    "events.csv",
    "timeline.csv",
    "tracking.csv",
]

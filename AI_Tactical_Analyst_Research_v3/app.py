"""AI Football Tactical Analyst - upload, analyze, and view dashboard.

This Streamlit app provides the complete product workflow:

    Upload video
        ↓
    Click Analyze
        ↓
    run_pipeline.py
        ↓
    run_full_analytics.py
        ↓
    Dashboard, annotated video, CSV and PDF downloads

Place this file in the project root as ``app.py``, beside:

- run_pipeline.py
- run_full_analytics.py
- models/
- src/
- outputs/

The app creates a separate folder for every uploaded match under:

    outputs/matches/<match_id>/

Important
---------
This is designed for local execution. The full YOLO and analytics pipeline may
be too heavy for free Streamlit Community Cloud resources.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "football_yolo11n.pt"
DEFAULT_MATCHES_DIR = PROJECT_ROOT / "outputs" / "matches"
DEFAULT_EXISTING_RUN = PROJECT_ROOT / "outputs" / "final_full_video"

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def configure_page() -> None:
    st.set_page_config(
        page_title="AI Football Tactical Analyst",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }

        .dashboard-title {
            font-size: 2.2rem;
            font-weight: 750;
            margin-bottom: 0.1rem;
        }

        .dashboard-subtitle {
            color: #6b7280;
            margin-bottom: 1rem;
        }

        .metric-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 14px;
            background: rgba(128, 128, 128, 0.05);
            min-height: 105px;
        }

        .metric-label {
            color: #6b7280;
            font-size: 0.85rem;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 4px;
        }

        .metric-note {
            color: #6b7280;
            font-size: 0.8rem;
            margin-top: 3px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "current_run_dir": str(DEFAULT_EXISTING_RUN),
        "analysis_running": False,
        "last_error": "",
        "last_log": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sanitize_filename(filename: str) -> str:
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    return safe[:60] or "uploaded_match"


def create_match_directory(filename: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_name = sanitize_filename(filename)
    match_dir = DEFAULT_MATCHES_DIR / f"{match_name}_{timestamp}"
    match_dir.mkdir(parents=True, exist_ok=False)
    return match_dir


def save_uploaded_video(uploaded_file, match_dir: Path) -> Path:
    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported video format. Use MP4, AVI, MOV, or MKV."
        )

    input_path = match_dir / f"input{extension}"
    input_path.write_bytes(uploaded_file.getbuffer())
    return input_path


def stream_subprocess(command: list[str]) -> Iterator[str]:
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None

    for line in iter(process.stdout.readline, ""):
        yield line.rstrip()

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}:\n"
            + " ".join(command)
        )


def run_stage(
    stage_name: str,
    command: list[str],
    log_placeholder,
    accumulated_logs: list[str],
) -> None:
    accumulated_logs.append(f"\n===== {stage_name} =====")

    for line in stream_subprocess(command):
        accumulated_logs.append(line)
        log_placeholder.code(
            "\n".join(accumulated_logs[-120:]),
            language="text",
        )


def analyze_uploaded_video(
    uploaded_file,
    model_path: Path,
    progress_placeholder,
    status_placeholder,
    log_placeholder,
) -> Path:
    if uploaded_file is None:
        raise ValueError("Please upload a football video first.")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    match_dir = create_match_directory(uploaded_file.name)
    input_path = save_uploaded_video(uploaded_file, match_dir)

    logs: list[str] = []

    pipeline_command = [
        sys.executable,
        str(PROJECT_ROOT / "run_pipeline.py"),
        "--video",
        str(input_path),
        "--model",
        str(model_path),
        "--output-dir",
        str(match_dir),
    ]

    analytics_command = [
        sys.executable,
        str(PROJECT_ROOT / "run_full_analytics.py"),
        "--run-dir",
        str(match_dir),
    ]

    status_placeholder.info("Stage 1/2: Running detection and tracking...")
    progress_placeholder.progress(10)

    run_stage(
        "Detection and Tracking",
        pipeline_command,
        log_placeholder,
        logs,
    )

    progress_placeholder.progress(58)
    status_placeholder.info("Stage 2/2: Running analytics and report generation...")

    run_stage(
        "Analytics and Reporting",
        analytics_command,
        log_placeholder,
        logs,
    )

    progress_placeholder.progress(100)
    status_placeholder.success("Analysis completed successfully.")

    st.session_state.last_log = "\n".join(logs)
    return match_dir


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as error:
        st.warning(f"Could not read {path.name}: {error}")
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        st.warning(f"Could not read {path.name}: {error}")
        return {}


def load_run_data(run_dir: Path) -> dict:
    analytics_dir = run_dir / "analytics"
    reports_dir = run_dir / "reports"

    return {
        "run_dir": run_dir,
        "input_video": find_input_video(run_dir),
        "annotated_video": run_dir / "annotated.mp4",
        "tracks": read_csv(run_dir / "tracks.csv"),
        "ball": read_csv(run_dir / "ball.csv"),
        "cleaned_tracks": read_csv(
            analytics_dir / "cleaned_tracks.csv"
        ),
        "player_statistics": read_csv(
            analytics_dir / "player_statistics.csv"
        ),
        "team_statistics": read_csv(
            analytics_dir / "team_statistics.csv"
        ),
        "tactical_timeline": read_csv(
            analytics_dir / "tactical_timeline.csv"
        ),
        "tactical_summary": read_csv(
            analytics_dir / "tactical_summary.csv"
        ),
        "possession_timeline": read_csv(
            analytics_dir / "possession_timeline_v2.csv"
        ),
        "possession_summary": read_csv(
            analytics_dir / "possession_summary_v2.csv"
        ),
        "transition_timeline": read_csv(
            analytics_dir / "transition_timeline.csv"
        ),
        "transition_summary": read_csv(
            analytics_dir / "transition_summary.csv"
        ),
        "pass_events": read_csv(
            analytics_dir / "pass_events.csv"
        ),
        "passing_edges": read_csv(
            analytics_dir / "passing_network_edges.csv"
        ),
        "passing_nodes": read_csv(
            analytics_dir / "passing_network_nodes.csv"
        ),
        "player_influence": read_csv(
            analytics_dir / "player_influence.csv"
        ),
        "player_rankings": read_csv(
            analytics_dir / "player_rankings.csv"
        ),
        "report_pdf": reports_dir / "match_report_detailed.pdf",
        "report_json_path": reports_dir / "match_summary_detailed.json",
        "report_json": read_json(
            reports_dir / "match_summary_detailed.json"
        ),
    }


def find_input_video(run_dir: Path) -> Path | None:
    candidates = [
        *run_dir.glob("input.mp4"),
        *run_dir.glob("input.avi"),
        *run_dir.glob("input.mov"),
        *run_dir.glob("input.mkv"),
    ]

    return candidates[0] if candidates else None


def render_metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_panel() -> None:
    st.subheader("Analyze a New Football Video")

    st.caption(
        "Upload a short football match clip. The application will run detection, "
        "tracking, analytics, heatmaps, and report generation."
    )

    upload_col, settings_col = st.columns([1.4, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            st.video(uploaded_file)

    with settings_col:
        model_text = st.text_input(
            "YOLO model path",
            value=str(DEFAULT_MODEL_PATH),
        )

        st.info(
            "The analysis can take several minutes on a CPU. "
            "Keep the browser and terminal open until it finishes."
        )

        analyze_button = st.button(
            "Analyze Video",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running,
        )

    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    log_expander = st.expander(
        "Processing Log",
        expanded=False,
    )

    with log_expander:
        log_placeholder = st.empty()

    if analyze_button:
        st.session_state.analysis_running = True
        st.session_state.last_error = ""

        try:
            run_dir = analyze_uploaded_video(
                uploaded_file=uploaded_file,
                model_path=Path(model_text),
                progress_placeholder=progress_placeholder,
                status_placeholder=status_placeholder,
                log_placeholder=log_placeholder,
            )

            st.session_state.current_run_dir = str(run_dir)
            st.success(
                f"Results saved to: {run_dir}"
            )

        except Exception as error:
            st.session_state.last_error = str(error)
            status_placeholder.error(
                f"Analysis failed: {error}"
            )

        finally:
            st.session_state.analysis_running = False


def render_overview(data: dict) -> None:
    tracks = data["cleaned_tracks"]
    player_stats = data["player_statistics"]
    possession = data["possession_summary"]
    transition = data["transition_summary"]

    tracking_rows = len(tracks)
    unique_tracks = (
        int(tracks["track_id"].nunique())
        if not tracks.empty and "track_id" in tracks
        else 0
    )
    players_scored = len(data["player_influence"])
    ball_frames = len(data["possession_timeline"])

    columns = st.columns(4)

    with columns[0]:
        render_metric_card(
            "Tracking rows",
            f"{tracking_rows:,}",
            "Cleaned frame-level observations",
        )
    with columns[1]:
        render_metric_card(
            "Unique tracks",
            str(unique_tracks),
            "Tracked object identities",
        )
    with columns[2]:
        render_metric_card(
            "Players scored",
            str(players_scored),
            "Clip-relative influence ratings",
        )
    with columns[3]:
        render_metric_card(
            "Ball frames",
            str(ball_frames),
            "Frames used in possession analysis",
        )

    st.markdown("### Annotated Video")

    if data["annotated_video"].exists():
        st.video(str(data["annotated_video"]))
    else:
        st.info("Annotated video has not been generated.")

    left, right = st.columns(2)

    with left:
        st.markdown("### Estimated Possession")

        if possession.empty:
            st.info("Possession summary is unavailable.")
        else:
            if {
                "team_id",
                "possession_pct_known",
            }.issubset(possession.columns):
                chart = possession.set_index("team_id")[
                    "possession_pct_known"
                ]
                st.bar_chart(chart)

            st.dataframe(
                possession,
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown("### Observed Transition")

        if transition.empty:
            st.info("Transition summary is unavailable.")
        else:
            row = transition.iloc[0]

            st.metric(
                "Dominant attacking team",
                str(
                    row.get(
                        "dominant_attacking_team",
                        "unknown",
                    )
                ),
            )
            st.metric(
                "Dominant transition phase",
                str(
                    row.get(
                        "dominant_transition_phase",
                        "unknown",
                    )
                ),
            )
            st.metric(
                "Analysis confidence",
                f"{100 * float(row.get('mean_analysis_confidence', 0.0)):.1f}%",
            )

    if player_stats.empty:
        st.warning("Player statistics were not generated for this run.")


def render_team_analytics(data: dict) -> None:
    team_stats = data["team_statistics"]
    tactical_summary = data["tactical_summary"]

    if team_stats.empty:
        st.info("Team statistics are unavailable.")
        return

    st.dataframe(
        team_stats,
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)

    with left:
        if {
            "team_id",
            "total_distance_m",
        }.issubset(team_stats.columns):
            st.markdown("### Total Team Distance")
            st.bar_chart(
                team_stats.set_index("team_id")[
                    "total_distance_m"
                ]
            )

    with right:
        if {
            "team_id",
            "average_speed_mps",
        }.issubset(team_stats.columns):
            st.markdown("### Average Team Speed")
            st.bar_chart(
                team_stats.set_index("team_id")[
                    "average_speed_mps"
                ]
            )

    st.markdown("### Tactical Shape Summary")

    if tactical_summary.empty:
        st.info("Tactical summary is unavailable.")
    else:
        st.dataframe(
            tactical_summary,
            use_container_width=True,
            hide_index=True,
        )


def render_tactical_timeline(data: dict) -> None:
    timeline = data["tactical_timeline"]

    if timeline.empty:
        st.info("Tactical timeline is unavailable.")
        return

    metric_options = {
        "Width": ("home_width_m", "away_width_m"),
        "Length": ("home_length_m", "away_length_m"),
        "Compactness": (
            "home_compactness_m",
            "away_compactness_m",
        ),
        "Centroid X": (
            "home_centroid_x_m",
            "away_centroid_x_m",
        ),
        "Visible Players": (
            "home_visible_players",
            "away_visible_players",
        ),
    }

    selected = st.selectbox(
        "Select tactical metric",
        list(metric_options.keys()),
    )

    selected_columns = [
        column
        for column in metric_options[selected]
        if column in timeline.columns
    ]

    if (
        "time_seconds" in timeline.columns
        and selected_columns
    ):
        st.line_chart(
            timeline.set_index("time_seconds")[
                selected_columns
            ]
        )

    st.dataframe(
        timeline,
        use_container_width=True,
        hide_index=True,
    )


def render_possession_transition(data: dict) -> None:
    possession = data["possession_timeline"]
    transition = data["transition_timeline"]

    left, right = st.columns(2)

    with left:
        st.markdown("### Possession Timeline")

        if possession.empty:
            st.info("Possession timeline is unavailable.")
        else:
            if "possession_team" in possession.columns:
                counts = (
                    possession["possession_team"]
                    .value_counts()
                    .rename_axis("team")
                    .to_frame("frames")
                )
                st.bar_chart(counts)

            st.dataframe(
                possession,
                use_container_width=True,
                hide_index=True,
            )

    with right:
        st.markdown("### Transition Timeline")

        if transition.empty:
            st.info("Transition timeline is unavailable.")
        else:
            chart_columns = [
                column
                for column in [
                    "centroid_progression_mps",
                    "analysis_confidence",
                ]
                if column in transition.columns
            ]

            if (
                "time_seconds" in transition.columns
                and chart_columns
            ):
                st.line_chart(
                    transition.set_index("time_seconds")[
                        chart_columns
                    ]
                )

            if "transition_phase" in transition.columns:
                phase_counts = (
                    transition["transition_phase"]
                    .value_counts()
                    .rename_axis("phase")
                    .to_frame("frames")
                )
                st.bar_chart(phase_counts)


def render_passing_network(data: dict) -> None:
    edges = data["passing_edges"]
    nodes = data["passing_nodes"]

    st.markdown("### Estimated Passing Edges")

    if edges.empty:
        st.info(
            "No teammate-to-teammate passing edge was inferred."
        )
    else:
        st.dataframe(
            edges,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Passing Network Nodes")

    if nodes.empty:
        st.info("Passing network nodes are unavailable.")
    else:
        st.dataframe(
            nodes,
            use_container_width=True,
            hide_index=True,
        )


def render_player_rankings(data: dict) -> None:
    rankings = data["player_rankings"]
    influence = data["player_influence"]

    if rankings.empty:
        st.info("Player rankings are unavailable.")
        return

    team_options = ["All"] + sorted(
        rankings["team_id"].dropna().astype(str).unique().tolist()
    )

    selected_team = st.selectbox(
        "Filter rankings by team",
        team_options,
    )

    filtered = rankings.copy()

    if selected_team != "All":
        filtered = filtered[
            filtered["team_id"] == selected_team
        ]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
    )

    if {
        "track_id",
        "influence_score",
    }.issubset(filtered.columns):
        top = filtered.head(12).copy()
        top.index = (
            "ID "
            + top["track_id"].astype(int).astype(str)
        )
        st.bar_chart(top["influence_score"])

    if influence.empty:
        return

    player_options = sorted(
        influence["track_id"].astype(int).unique().tolist()
    )

    selected_player = st.selectbox(
        "Inspect player track",
        player_options,
    )

    selected = influence[
        influence["track_id"].astype(int) == selected_player
    ]

    if selected.empty:
        return

    row = selected.iloc[0]
    columns = st.columns(4)

    with columns[0]:
        st.metric(
            "Influence score",
            f"{float(row.get('influence_score', 0.0)):.2f}",
        )
    with columns[1]:
        st.metric(
            "Distance",
            f"{float(row.get('total_distance_m', 0.0)):.2f} m",
        )
    with columns[2]:
        st.metric(
            "Average speed",
            f"{float(row.get('average_speed_mps', 0.0)):.2f} m/s",
        )
    with columns[3]:
        st.metric(
            "Possession time",
            f"{float(row.get('possession_time_sec', 0.0)):.2f} s",
        )


def draw_pitch(ax) -> None:
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")

    ax.plot(
        [0, 105, 105, 0, 0],
        [0, 0, 68, 68, 0],
    )
    ax.axvline(52.5)

    centre_circle = plt.Circle(
        (52.5, 34),
        9.15,
        fill=False,
    )
    ax.add_patch(centre_circle)

    ax.set_xlabel("Pitch length (m)")
    ax.set_ylabel("Pitch width (m)")


def render_heatmaps(data: dict) -> None:
    tracks = data["cleaned_tracks"]

    required = {
        "pitch_x_smooth_m",
        "pitch_y_smooth_m",
        "team_id",
        "track_id",
    }

    if tracks.empty or not required.issubset(tracks.columns):
        st.info("Pitch-coordinate tracking data is unavailable.")
        return

    heatmap_type = st.radio(
        "Heatmap type",
        ["Team", "Player"],
        horizontal=True,
    )

    if heatmap_type == "Team":
        teams = sorted(
            tracks["team_id"].dropna().astype(str).unique().tolist()
        )

        selected_team = st.selectbox(
            "Select team",
            teams,
        )

        subset = tracks[
            tracks["team_id"] == selected_team
        ]
        title = f"{selected_team} heatmap"

    else:
        player_ids = sorted(
            tracks["track_id"].astype(int).unique().tolist()
        )

        selected_player = st.selectbox(
            "Select player track",
            player_ids,
        )

        subset = tracks[
            tracks["track_id"].astype(int)
            == selected_player
        ]
        title = f"Player track {selected_player} heatmap"

    if subset.empty:
        st.info("No coordinates are available for this selection.")
        return

    figure, axis = plt.subplots(figsize=(10, 5.7))
    draw_pitch(axis)

    axis.hexbin(
        subset["pitch_x_smooth_m"],
        subset["pitch_y_smooth_m"],
        gridsize=22,
        mincnt=1,
    )

    axis.set_title(title)

    st.pyplot(
        figure,
        use_container_width=True,
    )
    plt.close(figure)


def render_downloads(data: dict) -> None:
    st.markdown("### Research Report")

    report_path = data["report_pdf"]

    if report_path.exists():
        st.download_button(
            "Download Detailed PDF Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Detailed PDF report is unavailable.")

    st.markdown("### Match Summary")

    if data["report_json"]:
        st.download_button(
            "Download Match Summary JSON",
            data=json.dumps(
                data["report_json"],
                indent=4,
            ),
            file_name="match_summary_detailed.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("### Analytics CSV Files")

    files = {
        "Player statistics": (
            data["player_statistics"],
            "player_statistics.csv",
        ),
        "Team statistics": (
            data["team_statistics"],
            "team_statistics.csv",
        ),
        "Tactical timeline": (
            data["tactical_timeline"],
            "tactical_timeline.csv",
        ),
        "Possession timeline": (
            data["possession_timeline"],
            "possession_timeline_v2.csv",
        ),
        "Transition timeline": (
            data["transition_timeline"],
            "transition_timeline.csv",
        ),
        "Passing network": (
            data["passing_edges"],
            "passing_network_edges.csv",
        ),
        "Player rankings": (
            data["player_rankings"],
            "player_rankings.csv",
        ),
    }

    columns = st.columns(2)

    for index, (label, (frame, filename)) in enumerate(
        files.items()
    ):
        with columns[index % 2]:
            if frame.empty:
                st.button(
                    f"{label} unavailable",
                    disabled=True,
                    use_container_width=True,
                )
            else:
                st.download_button(
                    f"Download {label}",
                    data=frame.to_csv(index=False),
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                )


def list_existing_runs() -> list[Path]:
    runs: list[Path] = []

    if DEFAULT_EXISTING_RUN.exists():
        runs.append(DEFAULT_EXISTING_RUN)

    if DEFAULT_MATCHES_DIR.exists():
        runs.extend(
            sorted(
                [
                    path
                    for path in DEFAULT_MATCHES_DIR.iterdir()
                    if path.is_dir()
                ],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    unique: list[Path] = []
    seen: set[str] = set()

    for run in runs:
        key = str(run.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(run)

    return unique


def main() -> None:
    configure_page()
    initialize_state()

    st.markdown(
        '<div class="dashboard-title">⚽ AI Football Tactical Analyst</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-subtitle">'
        "Upload a football video, analyze it automatically, and explore the results."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Match Selection")

        existing_runs = list_existing_runs()

        display_names = {
            str(path): (
                "Final Full Video"
                if path.resolve() == DEFAULT_EXISTING_RUN.resolve()
                else path.name
            )
            for path in existing_runs
        }

        current_value = st.session_state.current_run_dir

        options = [str(path) for path in existing_runs]

        if current_value and current_value not in options:
            options.insert(0, current_value)

        selected_run = st.selectbox(
            "Open analyzed match",
            options=options,
            format_func=lambda value: display_names.get(
                value,
                Path(value).name,
            ),
            index=(
                options.index(current_value)
                if current_value in options
                else 0
            )
            if options
            else None,
        )

        if selected_run:
            st.session_state.current_run_dir = selected_run

        st.caption(
            "Every uploaded video is stored in its own match folder."
        )

        st.divider()

        st.markdown("### Current Scope")
        st.info(
            "The system is optimized for short broadcast football clips. "
            "Possession, passes, transitions, and influence are estimates."
        )

    upload_tab, dashboard_tab = st.tabs(
        [
            "Upload & Analyze",
            "Analytics Dashboard",
        ]
    )

    with upload_tab:
        render_upload_panel()

    with dashboard_tab:
        current_run = Path(
            st.session_state.current_run_dir
        )

        st.caption(
            f"Loaded result folder: {current_run}"
        )

        if not current_run.exists():
            st.warning(
                "The selected result folder does not exist. "
                "Upload and analyze a video first."
            )
            return

        data = load_run_data(current_run)

        tabs = st.tabs(
            [
                "Overview",
                "Team Analytics",
                "Tactical Timeline",
                "Possession & Transition",
                "Passing Network",
                "Player Rankings",
                "Heatmaps",
                "Downloads",
            ]
        )

        with tabs[0]:
            render_overview(data)
        with tabs[1]:
            render_team_analytics(data)
        with tabs[2]:
            render_tactical_timeline(data)
        with tabs[3]:
            render_possession_transition(data)
        with tabs[4]:
            render_passing_network(data)
        with tabs[5]:
            render_player_rankings(data)
        with tabs[6]:
            render_heatmaps(data)
        with tabs[7]:
            render_downloads(data)


if __name__ == "__main__":
    main()
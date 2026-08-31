# AI_Tactical_Analyst_Research_v3

A complete research-style football tactical analysis system built in Python 3.11 for direct use in VS Code. The repository contains a runnable video-analysis pipeline, tactical board renderer, formation recognizer, event engine, Power BI-compatible exports, Streamlit dashboard, runtime dependency bootstrapper, calibration support, and helper scripts.

## Features

- Ultralytics YOLO11-based football entity detection supporting `player`, `goalkeeper`, `referee`, `ball`, and optional `coach`
- ByteTrack-inspired multi-object tracking with separate ball handling, lost-track recovery, and stable IDs
- Research-oriented team classification using shirt crop extraction, HSV/LAB features, KMeans, grass filtering, outlier removal, temporal smoothing, and majority voting
- Automatic homography estimation plus manual calibration support with persisted calibration JSON
- Tactical board / mini-map overlay with player IDs, formations, and ball location
- Formation recognition for `4-4-2`, `4-3-3`, `4-2-3-1`, `3-5-2`, `3-4-3`, `4-1-4-1`, `4-5-1`, `5-3-2`, and `5-4-1`
- Heatmaps for players, teams, ball, and possession
- Player statistics including distance, speed, sprint count, touches, possession, time by half, recoveries, and pass estimation
- Heuristic event detection for pass, interception, ball recovery, shot, goal attempt, corner, throw, goal kick, and offside
- Annotated MP4 export with live labels, FPS, ball trail, and tactical-board inset
- Streamlit dashboard with upload, run, preview, analytics tables, charts, downloads, and heatmaps
- Power BI-ready UTF-8 CSV exports plus PDF implementation guide
- Runtime package bootstrapper for missing dependencies

## Repository Layout

```text
AI_Tactical_Analyst_Research_v3/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── configs/
│   ├── default.yaml
│   └── calibration_sample.json
├── models/
│   └── README.md
├── outputs/
│   ├── heatmaps/
│   └── reports/
├── powerbi/
├── scripts/
│   ├── bootstrap.py
│   ├── create_sample_video.py
│   └── generate_powerbi_guide.py
└── src/
    ├── analytics/
    ├── calibration/
    ├── classification/
    ├── core/
    ├── detection/
    ├── tracking/
    ├── utils/
    └── visualization/
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the Streamlit dashboard

```bash
streamlit run app.py
```

### 3) Or run the CLI pipeline

```bash
python run_pipeline.py --video /path/to/match.mp4
```

### 4) Optional smoke test

Generate a small synthetic clip:

```bash
python scripts/create_sample_video.py
python run_pipeline.py --video outputs/sample_demo.mp4 --max-frames 120
```

## Model Support

The system supports the following model paths:

- `models/yolo11n.pt`
- `models/football_yolo11n.pt`

You may override the path from the Streamlit UI or CLI. If local weights are missing, the detector will fall back to the generic YOLO11 path and then defer to Ultralytics' automatic download mechanism.

## Calibration

- `configs/calibration_sample.json` contains example correspondences
- Automatic homography estimates a pitch quadrilateral from the green field mask
- Manual calibration provides stronger geometric stability for tactical-board outputs

## Generated Outputs

Primary outputs are written into `outputs/`:

- `annotated.mp4`
- `tracks.csv`
- `tracking.csv`
- `ball.csv`
- `events.csv`
- `formations.csv`
- `timeline.csv`
- `player_stats.csv`
- `heatmaps/*.png`
- `reports/summary_report.md`
- `powerbi/PowerBI_Dashboard_Guide.pdf`

## Power BI Integration

Use `powerbi/PowerBI_Dashboard_Guide.pdf` for import steps, relationship design, measures, KPIs, charts, and example DAX. CSV files are exported with UTF-8 BOM to remain Power BI friendly on Windows systems.

## Notes on Practical Use

- Football-specific detection quality depends heavily on the checkpoint used.
- For high-quality deployment, replace `models/football_yolo11n.pt` with a fine-tuned broadcast-football model.
- Automatic homography is intentionally conservative; use manual calibration for research-grade metric accuracy.
- Event detection is heuristic and intended for tactical acceleration rather than officiating.

## References

- Ultralytics object tracking documentation: https://docs.ultralytics.com/modes/track
- Ultralytics YOLO11 documentation: https://docs.ultralytics.com/models/yolo11
- Supervision tracking guide: https://supervision.roboflow.com/how_to/track_objects/
- Streamlit download button docs: https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
- Streamlit theming docs: https://docs.streamlit.io/develop/concepts/configuration/theming

# Models

Supported model paths:

- `models/yolo11n.pt`
- `models/football_yolo11n.pt`

The pipeline accepts either path from the config or Streamlit UI. If a requested local file is missing, the detector falls back to `models/yolo11n.pt` and then to Ultralytics' automatic download behavior.

Place your fine-tuned football weights here before running production workloads.

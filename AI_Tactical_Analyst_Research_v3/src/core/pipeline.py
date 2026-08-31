"""End-to-end tactical analysis pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.analytics.events import EventDetector
from src.analytics.formation import FormationRecognizer
from src.analytics.heatmaps import HeatmapEngine
from src.analytics.stats import StatisticsEngine
from src.calibration.homography import HomographyEstimator
from src.core.config import AppConfig
from src.core.io import ensure_project_dirs, write_csv
from src.core.types import EventRecord, FormationSnapshot, TrackState
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTracker
from src.utils.geometry import bottom_center, perspective_transform
from src.utils.video import VideoReader, VideoWriter
from src.visualization.annotator import VideoAnnotator
from src.visualization.reports import ReportGenerator
from src.classification.team_classifier import TeamClassifier


class TacticalAnalysisPipeline:
    """Professional research-style football tactical analysis pipeline."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.fps = 25.0
        ensure_project_dirs()

    def run(
        self,
        video_path: str | Path,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> dict[str, Path]:
        """Run the full pipeline on a single video."""
        reader = VideoReader(video_path)
        metadata = reader.metadata
        self.fps = metadata.fps
        detector = YoloDetector(self.config.detector, self.logger)
        tracker = ByteTracker(self.config.tracker, metadata.fps, self.logger)
        classifier = TeamClassifier(self.config.team_classifier, self.logger)
        homography = HomographyEstimator(self.config.homography, self.logger)
        formation = FormationRecognizer(self.config.formation, self.logger)
        heatmaps = HeatmapEngine(self.config.heatmap, self.logger)
        stats_engine = StatisticsEngine(metadata.fps, self.logger)
        event_detector = EventDetector(self.config.events, metadata.fps, self.logger)
        annotator = VideoAnnotator() if self.config.annotate_video else None
        report_generator = ReportGenerator(self.logger)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = output_dir / "annotated.mp4"
        writer = VideoWriter(annotated_path, metadata.width, metadata.height, metadata.fps) if self.config.export_annotated_video else None

        track_rows: list[dict] = []
        tracking_rows: list[dict] = []
        ball_rows: list[dict] = []
        event_rows: list[dict] = []
        formation_rows: list[dict] = []
        timeline_rows: list[dict] = []

        t0 = time.time()
        processed_frames = 0
        try:
            for frame_index, frame in reader:
                if frame_index % self.config.video_stride != 0:
                    continue
                if self.config.max_frames is not None and processed_frames >= self.config.max_frames:
                    break
                processed_frames += 1
                detections = detector.infer(frame, frame_index)
                homography_state = homography.maybe_refresh(frame, frame_index)
                self._attach_pitch_coordinates(detections, homography_state.matrix)
                tracks, ball_track = tracker.update(detections, frame_index)
                if frame_index < 10:
                    print(f"\nFRAME {frame_index}")
                    print("Detections:", len(detections))
                    print("Tracks:", len(tracks))
                    print([d.class_name for d in detections])
                self._propagate_pitch_coordinates_to_tracks(tracks, ball_track, homography_state.matrix)
                classifier.classify(frame, tracks)
                formations = formation.update(frame_index, tracks)
                events, possessing_track_id = event_detector.update(frame_index, tracks, ball_track)
                stats_engine.update_tracks(tracks, possessing_track_id)
                stats_engine.register_events(events)
                heatmaps.update(tracks, ball_track, possessing_track_id)
                self._collect_rows(track_rows, tracking_rows, ball_rows, event_rows, formation_rows, timeline_rows, frame_index, tracks, ball_track, events, formations, metadata.fps)
                if writer is not None and annotator is not None:
                    annotated = annotator.draw(frame, tracks, ball_track, formations, events, frame_index, metadata.fps)
                    writer.write(annotated)
                if progress_callback:
                    progress = frame_index / max(1, metadata.frame_count)
                    progress_callback(progress, f"Processed frame {frame_index}/{metadata.frame_count}")
        finally:
            reader.release()
            if writer is not None:
                writer.release()

        player_stats = stats_engine.to_frame()
        heatmap_paths = heatmaps.render_all()

        outputs = {
            "annotated_video": annotated_path,
            "tracks": output_dir / "tracks.csv",
            "tracking": output_dir / "tracking.csv",
            "ball": output_dir / "ball.csv",
            "events": output_dir / "events.csv",
            "formations": output_dir / "formations.csv",
            "timeline": output_dir / "timeline.csv",
            "player_stats": output_dir / "player_stats.csv",
        }

        write_csv(outputs["tracks"], pd.DataFrame(track_rows))
        write_csv(outputs["tracking"], pd.DataFrame(tracking_rows))
        write_csv(outputs["ball"], pd.DataFrame(ball_rows))
        write_csv(outputs["events"], pd.DataFrame(event_rows))
        write_csv(outputs["formations"], pd.DataFrame(formation_rows))
        write_csv(outputs["timeline"], pd.DataFrame(timeline_rows))
        write_csv(outputs["player_stats"], player_stats)
        outputs["powerbi_guide"] = report_generator.generate_powerbi_guide()
        outputs["summary_report"] = report_generator.write_summary_markdown(outputs, player_stats)

        self.logger.info("Pipeline completed in %.2fs across %d frames", time.time() - t0, processed_frames)
        if progress_callback:
            progress_callback(1.0, "Pipeline complete")
        return outputs | {f"heatmap_{idx}": path for idx, path in enumerate(heatmap_paths, start=1)}

    def _attach_pitch_coordinates(self, detections, homography_matrix: np.ndarray | None) -> None:
        points = [det.foot_point if det.class_name != "ball" else det.center for det in detections]
        transformed = perspective_transform(points, homography_matrix)
        for det, pitch_xy in zip(detections, transformed, strict=False):
            det.pitch_xy = pitch_xy

    def _propagate_pitch_coordinates_to_tracks(self, tracks: list[TrackState], ball_track: TrackState | None, homography_matrix: np.ndarray | None) -> None:
        all_tracks = list(tracks)
        if ball_track is not None:
            all_tracks.append(ball_track)
        points = [bottom_center(track.bbox_xyxy) if track.class_name != "ball" else ((track.bbox_xyxy[0] + track.bbox_xyxy[2]) / 2.0, (track.bbox_xyxy[1] + track.bbox_xyxy[3]) / 2.0) for track in all_tracks]
        transformed = perspective_transform(points, homography_matrix)
        for track, pitch_xy in zip(all_tracks, transformed, strict=False):
            track.pitch_xy = pitch_xy
            if pitch_xy is not None and (not track.history_pitch_xy or track.history_pitch_xy[-1] != pitch_xy):
                track.history_pitch_xy.append(pitch_xy)
                track.history_pitch_xy = track.history_pitch_xy[-300:]
            self._update_track_speed(track)

    def _update_track_speed(self, track: TrackState) -> None:
        if len(track.history_pitch_xy) < 2:
            track.smoothed_speed_mps = 0.0
            return
        delta = np.array(track.history_pitch_xy[-1]) - np.array(track.history_pitch_xy[-2])
        speed = float(np.linalg.norm(delta)) * max(self.fps, 1.0)
        track.smoothed_speed_mps = 0.7 * track.smoothed_speed_mps + 0.3 * speed

    def _collect_rows(
        self,
        track_rows: list[dict],
        tracking_rows: list[dict],
        ball_rows: list[dict],
        event_rows: list[dict],
        formation_rows: list[dict],
        timeline_rows: list[dict],
        frame_index: int,
        tracks: list[TrackState],
        ball_track: TrackState | None,
        events: list[EventRecord],
        formations: list[FormationSnapshot],
        fps: float,
    ) -> None:
        time_seconds = frame_index / fps
        for track in tracks:
            row = {
                "frame_index": frame_index,
                "time_seconds": round(time_seconds, 3),
                "track_id": track.track_id,
                "class_name": track.class_name,
                "team_id": track.team_id,
                "bbox_x1": round(float(track.bbox_xyxy[0]), 2),
                "bbox_y1": round(float(track.bbox_xyxy[1]), 2),
                "bbox_x2": round(float(track.bbox_xyxy[2]), 2),
                "bbox_y2": round(float(track.bbox_xyxy[3]), 2),
                "pitch_x_m": round(float(track.pitch_xy[0]), 3) if track.pitch_xy else None,
                "pitch_y_m": round(float(track.pitch_xy[1]), 3) if track.pitch_xy else None,
                "speed_mps": round(track.smoothed_speed_mps, 3),
                "confidence": round(track.confidence, 4),
            }
            track_rows.append(row)
            tracking_rows.append(row | {"history_length": len(track.history_pitch_xy), "hits": track.hits, "missed": track.missed})
        if ball_track is not None:
            ball_rows.append(
                {
                    "frame_index": frame_index,
                    "time_seconds": round(time_seconds, 3),
                    "track_id": ball_track.track_id,
                    "bbox_x1": round(float(ball_track.bbox_xyxy[0]), 2),
                    "bbox_y1": round(float(ball_track.bbox_xyxy[1]), 2),
                    "bbox_x2": round(float(ball_track.bbox_xyxy[2]), 2),
                    "bbox_y2": round(float(ball_track.bbox_xyxy[3]), 2),
                    "pitch_x_m": round(float(ball_track.pitch_xy[0]), 3) if ball_track.pitch_xy else None,
                    "pitch_y_m": round(float(ball_track.pitch_xy[1]), 3) if ball_track.pitch_xy else None,
                    "speed_mps": round(ball_track.smoothed_speed_mps, 3),
                    "confidence": round(ball_track.confidence, 4),
                }
            )
        for event in events:
            event_rows.append(
                {
                    "frame_index": event.frame_index,
                    "time_seconds": round(time_seconds, 3),
                    "event_type": event.event_type,
                    "team_id": event.team_id,
                    "primary_track_id": event.primary_track_id,
                    "secondary_track_id": event.secondary_track_id,
                    "ball_track_id": event.ball_track_id,
                    "start_pitch_x_m": round(float(event.start_pitch_xy[0]), 3) if event.start_pitch_xy else None,
                    "start_pitch_y_m": round(float(event.start_pitch_xy[1]), 3) if event.start_pitch_xy else None,
                    "end_pitch_x_m": round(float(event.end_pitch_xy[0]), 3) if event.end_pitch_xy else None,
                    "end_pitch_y_m": round(float(event.end_pitch_xy[1]), 3) if event.end_pitch_xy else None,
                    "confidence": round(event.confidence, 4),
                    "details_json": str(event.details),
                }
            )
        for snapshot in formations:
            formation_rows.append(
                {
                    "frame_index": snapshot.frame_index,
                    "time_seconds": round(time_seconds, 3),
                    "team_id": snapshot.team_id,
                    "formation": snapshot.formation,
                    "line_counts": "-".join(map(str, snapshot.line_counts)),
                    "direction": snapshot.direction,
                    "confidence": round(snapshot.confidence, 4),
                }
            )
        timeline_rows.append(
            {
                "frame_index": frame_index,
                "time_seconds": round(time_seconds, 3),
                "num_tracks": len(tracks),
                "ball_visible": ball_track is not None,
                "num_events": len(events),
                "formations_detected": "; ".join(f"{f.team_id}:{f.formation}" for f in formations),
            }
        )

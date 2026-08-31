"""Ultralytics YOLO11 detector wrapper for football entities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.core.config import DetectorConfig
from src.core.constants import CLASS_NAMES, MODEL_DIR
from src.core.types import Detection
from src.utils.geometry import clip_box

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency bootstrap handles install
    YOLO = None


class YoloDetector:
    """YOLO11 detector with football-specific postprocessing.

    The class supports custom football checkpoints and generic YOLO11 models.
    If the requested model path does not exist, the loader falls back to a
    bundled generic model path so the project still remains runnable.
    """

    def __init__(self, config: DetectorConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.model = self._load_model(config.model_path)
        self.model_name = Path(config.model_path).name

    def _load_model(self, model_path: str) -> YOLO:
        if YOLO is None:
            raise ImportError("ultralytics is required to use YoloDetector")
        requested = Path(model_path)
        fallback = MODEL_DIR / "yolo11n.pt"
        if requested.exists():
            resolved = requested
        elif fallback.exists():
            self.logger.warning("Requested model %s not found; falling back to %s", requested, fallback)
            resolved = fallback
        else:
            self.logger.warning("No local YOLO weights found. Ultralytics will download %s on first run.", requested.name)
            resolved = requested
        return YOLO(str(resolved))

    def infer(self, frame: np.ndarray, frame_index: int) -> list[Detection]:
        """Run detection on one frame and return typed detections."""
        results = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            device=self.config.device,
            half=self.config.half_precision,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        image_h, image_w = frame.shape[:2]
        for idx in range(len(boxes)):
            box = boxes[idx]
            cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            if cls_id == 3 and conf < self.config.ball_confidence:
                continue
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
            xyxy = clip_box(xyxy, image_w, image_h)
            raw_names = result.names if hasattr(result, "names") else {}

            if isinstance(raw_names, dict):
                class_name = raw_names.get(cls_id, str(cls_id))
            elif isinstance(raw_names, (list, tuple)):
                class_name = raw_names[cls_id]
            else:
                class_name = str(cls_id)

            detections.append(
                Detection(
                    bbox_xyxy=xyxy,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                    frame_index=frame_index,
                    model_name=self.model_name,
                    metadata={
                        "raw_class_name": class_name,
                    },
                )
            )
        return self._deduplicate_entities(detections)

    def _deduplicate_entities(self, detections: list[Detection]) -> list[Detection]:
        """Suppress near-identical boxes conservatively across classes."""
        kept: list[Detection] = []
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        for det in detections:
            duplicate = False
            for current in kept:
                iou = self._iou(current.bbox_xyxy, det.bbox_xyxy)
                if iou > 0.85 and current.class_id == det.class_id:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

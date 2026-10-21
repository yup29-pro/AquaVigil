"""
backend/app/services/detection.py

Inference service for AquaVigil.
Safely checks for trained model checkpoints and provides structured detection handling.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import time

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.schemas import DetectionItem, BoundingBox


class FishDiseaseDetectionService:
    def __init__(self, weights_path: str = "models/best.pt"):
        self.weights_path = Path(weights_path)
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads YOLOv8 weights if checkpoint exists and Ultralytics is installed."""
        if YOLO is None:
            print("[FishDiseaseDetectionService] Ultralytics is not installed.")
            return

        if self.weights_path.exists():
            print(f"[FishDiseaseDetectionService] Loading model from {self.weights_path.resolve()}...")
            self.model = YOLO(str(self.weights_path))
        else:
            print(f"[FishDiseaseDetectionService] Checkpoint not found at {self.weights_path.resolve()}. Service operating in standby mode.")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def run_detection(self, image_bytes: bytes, conf_thresh: float = 0.25) -> Tuple[List[DetectionItem], float]:
        """
        Runs inference on provided image bytes.
        Returns a tuple of (detections_list, latency_in_ms).
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Model checkpoint '{self.weights_path}' is not yet available. "
                "AquaVigil cannot perform real detections until a model is trained in Phase 2."
            )

        import numpy as np
        import cv2

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file provided.")

        start_time = time.time()
        results = self.model.predict(source=img, conf=conf_thresh, verbose=False)
        latency_ms = (time.time() - start_time) * 1000.0

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id] if self.model.names else str(cls_id)
                confidence = float(box.conf[0])
                coords = [float(c) for c in box.xyxy[0]]
                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(confidence, 4),
                        box=BoundingBox(
                            x1=round(coords[0], 2),
                            y1=round(coords[1], 2),
                            x2=round(coords[2], 2),
                            y2=round(coords[3], 2),
                        ),
                    )
                )

        return detections, round(latency_ms, 2)

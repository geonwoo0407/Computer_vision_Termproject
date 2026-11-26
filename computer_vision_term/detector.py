"""YOLO detector and tracking-module-friendly result serialization."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Iterable

import cv2
import numpy as np

from camera import CameraIntrinsics
from depth import estimate_object_depth

CLASS_COLORS_BGR = {"person": (60, 210, 90), "chair": (245, 170, 70)}


class DetectorDependencyError(RuntimeError):
    """Raised when the optional YOLO runtime is not installed."""


class VisionDetector:
    """Detect the configured object classes in one RGB frame.

    This class deliberately does not assign tracking IDs or decide seat state.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "cpu",
        target_classes: Iterable[str] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except (ImportError, OSError) as exc:
            raise DetectorDependencyError(
                "Ultralytics YOLO is not available. Install project dependencies "
                "with: python -m pip install -r requirements.txt"
            ) from exc

        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in the range (0, 1]")
        if image_size <= 0:
            raise ValueError("image_size must be positive")

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

        names = self.model.names
        self.class_names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        requested = set(target_classes or self.class_names.values())
        self.class_ids = [
            class_id
            for class_id, class_name in self.class_names.items()
            if class_name in requested
        ]
        unsupported = requested.difference(self.class_names.values())
        if unsupported:
            unsupported_text = ", ".join(sorted(unsupported))
            raise ValueError(f"Model does not support configured classes: {unsupported_text}")

    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: float,
        depth_m: np.ndarray | None = None,
        intrinsics: CameraIntrinsics | None = None,
        *,
        native_intrinsics: object | None = None,
        serial_number: str | None = None,
        frame_id: int | None = None,
        color_timestamp_ms: float | None = None,
        depth_timestamp_ms: float | None = None,
    ) -> dict[str, Any]:
        """Run detection and return JSON-serializable observations."""
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")

        started = time.perf_counter()
        predictions = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        detections: list[dict[str, Any]] = []
        boxes = predictions[0].boxes
        if boxes is not None:
            xyxy_values = boxes.xyxy.detach().cpu().numpy()
            confidence_values = boxes.conf.detach().cpu().numpy()
            class_values = boxes.cls.detach().cpu().numpy().astype(int)

            for xyxy, score, class_id in zip(
                xyxy_values, confidence_values, class_values
            ):
                x1, y1, x2, y2 = (int(round(value)) for value in xyxy.tolist())
                object_depth = None
                position_3d = None
                valid_depth_pixels = 0
                if depth_m is not None:
                    object_depth, position_3d, valid_depth_pixels = (
                        estimate_object_depth(
                            depth_m,
                            [x1, y1, x2, y2],
                            intrinsics,
                            native_intrinsics=native_intrinsics,
                        )
                    )
                detections.append(
                    {
                        "class_id": int(class_id),
                        "class_name": self.class_names[int(class_id)],
                        "confidence": round(float(score), 4),
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                        "depth_m": object_depth,
                        "position_3d_m": position_3d,
                        "valid_depth_pixels": valid_depth_pixels,
                        "seat_candidates": [],
                    }
                )

        height, width = frame.shape[:2]
        return {
            "schema_version": "1.0",
            "camera_id": camera_id,
            "serial_number": serial_number,
            "frame_id": frame_id,
            "timestamp": float(timestamp),
            "timestamp_ms": float(timestamp * 1000.0),
            "color_timestamp_ms": color_timestamp_ms,
            "depth_timestamp_ms": depth_timestamp_ms,
            "coordinate_system": f"{camera_id}_color_optical_frame",
            "image_width": width,
            "image_height": height,
            "frame_size": {"width": width, "height": height},
            "depth_available": depth_m is not None,
            "camera_intrinsics": asdict(intrinsics) if intrinsics else None,
            "processing_time_ms": round(inference_ms, 2),
            "detections": detections,
        }

    @staticmethod
    def draw_detections(
        frame: np.ndarray,
        result: dict[str, Any],
        fps: float | None = None,
    ) -> np.ndarray:
        """Draw a compact debugging overlay on a copy of the frame."""
        output = frame.copy()
        for detection in result["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            color = CLASS_COLORS_BGR.get(class_name, (40, 170, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            if detection.get("depth_m") is not None:
                label += f" {detection['depth_m']:.2f}m"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y = max(y1, text_height + 8)
            cv2.rectangle(
                output,
                (x1, label_y - text_height - 8),
                (x1 + text_width + 6, label_y),
                color,
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 3, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        status = result["camera_id"]
        if fps is not None:
            status += f" | FPS {fps:.1f}"
        cv2.putText(
            output,
            status,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output


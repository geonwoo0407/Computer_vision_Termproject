"""Seat ROI loading, object overlap candidates, and observation summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SeatROIManager:
    def __init__(self, config_path: str, min_overlap: float = 0.05) -> None:
        path = Path(config_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Seat ROI file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid seat ROI JSON: {path} ({exc})") from exc

        image_size = data.get("image_size", {})
        self.reference_width = int(image_size.get("width", 0))
        self.reference_height = int(image_size.get("height", 0))
        if self.reference_width <= 0 or self.reference_height <= 0:
            raise ValueError("Seat ROI image_size width and height must be positive")
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_overlap must be in the range [0, 1]")

        self.calibrated = bool(data.get("calibrated", False))
        self.min_overlap = min_overlap
        self.seats: list[dict[str, Any]] = []
        seat_ids: set[str] = set()
        for seat in data.get("seats", []):
            seat_id = str(seat["seat_id"])
            if seat_id in seat_ids:
                raise ValueError(f"Duplicate seat_id in ROI config: {seat_id}")
            seat_ids.add(seat_id)

            if "rectangle" in seat:
                x1, y1, x2, y2 = seat["rectangle"]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            else:
                points = seat.get("polygon", [])
            polygon = np.asarray(points, dtype=np.float32)
            if polygon.shape[0] < 3 or polygon.shape[1:] != (2,):
                raise ValueError(f"Seat {seat_id} needs a rectangle or polygon")
            if not cv2.isContourConvex(polygon):
                raise ValueError(f"Seat {seat_id} polygon must be convex")
            self.seats.append({"seat_id": seat_id, "polygon": polygon})

    def add_candidates(
        self,
        detections: list[dict[str, Any]],
        frame_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Attach ROI/bbox intersection ratios and return per-seat observations."""
        if not self.calibrated:
            return []

        width, height = frame_size
        scaled_seats = self._scaled_seats(width, height)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            bbox_width = max(0, x2 - x1)
            bbox_height = max(0, y2 - y1)
            bbox_area = float(bbox_width * bbox_height)
            candidates = []
            if bbox_area > 0:
                bbox_polygon = np.asarray(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
                for seat in scaled_seats:
                    intersection, _ = cv2.intersectConvexConvex(
                        bbox_polygon, seat["polygon"]
                    )
                    ratio = float(intersection) / bbox_area
                    if ratio >= self.min_overlap:
                        candidates.append(
                            {
                                "seat_id": seat["seat_id"],
                                "overlap_ratio": round(ratio, 4),
                            }
                        )
            detection["seat_candidates"] = sorted(
                candidates, key=lambda item: item["overlap_ratio"], reverse=True
            )

        observations = []
        for seat in scaled_seats:
            seat_id = seat["seat_id"]
            matched = [
                (index, detection)
                for index, detection in enumerate(detections)
                if any(
                    candidate["seat_id"] == seat_id
                    for candidate in detection["seat_candidates"]
                )
            ]
            observations.append(
                {
                    "seat_id": seat_id,
                    "person_detected": any(
                        detection["class_name"] == "person"
                        for _, detection in matched
                    ),
                    "belonging_classes": sorted(
                        {
                            detection["class_name"]
                            for _, detection in matched
                            if detection["class_name"] not in ("person", "chair")
                        }
                    ),
                    "detection_indices": [index for index, _ in matched],
                }
            )
        return observations

    def draw(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        color = (80, 220, 255) if self.calibrated else (80, 80, 255)
        for seat in self._scaled_seats(width, height):
            polygon = np.rint(seat["polygon"]).astype(np.int32)
            cv2.polylines(frame, [polygon], True, color, 2)
            anchor = tuple(polygon[0].tolist())
            label = f"Seat {seat['seat_id']}"
            if not self.calibrated:
                label += " (setup)"
            cv2.putText(
                frame,
                label,
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def _scaled_seats(self, width: int, height: int) -> list[dict[str, Any]]:
        scale = np.asarray(
            [width / self.reference_width, height / self.reference_height],
            dtype=np.float32,
        )
        return [
            {"seat_id": seat["seat_id"], "polygon": seat["polygon"] * scale}
            for seat in self.seats
        ]

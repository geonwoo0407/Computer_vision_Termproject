"""Robust depth extraction for an object bounding box."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class IntrinsicsLike(Protocol):
    fx: float
    fy: float
    ppx: float
    ppy: float


def estimate_object_depth(
    depth_m: np.ndarray,
    bbox: list[int],
    intrinsics: IntrinsicsLike | None,
    center_ratio: float = 0.5,
    min_valid_pixels: int = 20,
    min_depth_m: float = 0.1,
    max_depth_m: float = 10.0,
    native_intrinsics: object | None = None,
) -> tuple[float | None, list[float] | None, int]:
    """Return median central depth, camera-space XYZ, and valid pixel count."""
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 <= x1 or y2 <= y1:
        return None, None, 0

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_width = max(1, int((x2 - x1) * center_ratio))
    crop_height = max(1, int((y2 - y1) * center_ratio))
    crop_x1 = max(0, int(round(center_x - crop_width / 2)))
    crop_x2 = min(width, int(round(center_x + crop_width / 2)))
    crop_y1 = max(0, int(round(center_y - crop_height / 2)))
    crop_y2 = min(height, int(round(center_y + crop_height / 2)))

    values = depth_m[crop_y1:crop_y2, crop_x1:crop_x2]
    valid = values[
        np.isfinite(values) & (values >= min_depth_m) & (values <= max_depth_m)
    ]
    valid_count = int(valid.size)
    if valid_count < min_valid_pixels:
        return None, None, valid_count

    median_depth = float(np.median(valid))
    position = None
    if native_intrinsics is not None:
        try:
            import pyrealsense2 as rs

            point = rs.rs2_deproject_pixel_to_point(
                native_intrinsics,
                [float(center_x), float(center_y)],
                median_depth,
            )
            position = [round(float(value), 4) for value in point]
        except (ImportError, OSError, RuntimeError, TypeError):
            position = None
    elif intrinsics is not None and intrinsics.fx > 0 and intrinsics.fy > 0:
        x_m = (center_x - intrinsics.ppx) / intrinsics.fx * median_depth
        y_m = (center_y - intrinsics.ppy) / intrinsics.fy * median_depth
        position = [round(float(x_m), 4), round(float(y_m), 4), round(median_depth, 4)]

    return round(median_depth, 4), position, valid_count

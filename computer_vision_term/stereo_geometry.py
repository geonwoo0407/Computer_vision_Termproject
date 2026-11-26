"""Metric geometry of two RGB optical frames. Camera 1 is the common frame."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from camera import CameraIntrinsics


def camera_matrix(intr):
    return np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], np.float64)


def ideal_pixels(intr, pixels):
    """Return undistorted pixels at the original focal length, suitable for OpenCV."""
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
    model = intr.distortion_model.split(".")[-1]
    coeffs = np.asarray(intr.coefficients, np.float64)
    brown_models = ("brown_conrady", "modified_brown_conrady", "inverse_brown_conrady")
    if model == "none" or (model in brown_models and not coeffs.any()):
        return points.copy()
    k = camera_matrix(intr)
    if model == "brown_conrady":
        return cv2.undistortPoints(points, k, coeffs, P=k)
    if model == "inverse_brown_conrady":
        # 역방향 계수를 OpenCV의 순방향 왜곡 계수로 넘기면 안 된다.
        import pyrealsense2 as rs
        native = rs.intrinsics()
        native.width, native.height = intr.width, intr.height
        native.fx, native.fy, native.ppx, native.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        native.model, native.coeffs = rs.distortion.inverse_brown_conrady, list(intr.coefficients)
        rays = [rs.rs2_deproject_pixel_to_point(native, p.tolist(), 1.) for p in points[:, 0]]
        return np.asarray(rays)[:, None, :2] * [intr.fx, intr.fy] + [intr.ppx, intr.ppy]
    raise ValueError(f"Unsupported RGB distortion model {model} for stereo calibration; check the color profile.")


def board_points(pattern, square_mm):
    cols, rows = pattern
    if min(cols, rows) < 3 or not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Use at least 3x3 inner corners and a positive measured square size in mm.")
    points = np.zeros((cols*rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * (square_mm / 1000)
    return points


def validate_transform(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Expected a 4x4 cam2-to-cam1 matrix.")
    r = matrix[:3, :3]
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5)
            or np.linalg.norm(matrix[:3, 3]) < 1e-6):
        raise ValueError("Invalid cam2-to-cam1 rigid transform.")


def calculate_3d_distance(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


@dataclass
class StereoCalibration:
    serials: tuple
    intrinsics: dict
    cam1_from_cam2: np.ndarray
    metadata: dict

    def __post_init__(self):
        if len(self.serials) != 2 or len(set(self.serials)) != 2 or set(self.serials) != set(self.intrinsics):
            raise ValueError("Expected two distinct camera serials with matching RGB profiles.")
        for intr in self.intrinsics.values():
            values = [intr.fx, intr.fy, intr.ppx, intr.ppy, *intr.coefficients]
            if (min(intr.width, intr.height, intr.fx, intr.fy) <= 0 or len(intr.coefficients) != 5
                    or not np.isfinite(values).all()):
                raise ValueError("Invalid RGB intrinsics in calibration.")
        validate_transform(self.cam1_from_cam2)

    def get_camera_intrinsics(self, serial):
        if serial not in self.intrinsics:
            raise ValueError(f"Camera {serial} is not in this stereo calibration.")
        return self.intrinsics[serial]

    def check_profile(self, serial, intr, fps):
        saved = self.get_camera_intrinsics(serial)
        if ((saved.width, saved.height, saved.distortion_model) != (intr.width, intr.height, intr.distortion_model)
                or not np.allclose(camera_matrix(saved), camera_matrix(intr))
                or not np.allclose(saved.coefficients, intr.coefficients)
                or not np.isclose(self.metadata["fps"][serial], fps)):
            raise ValueError(f"RGB profile/FPS changed for {serial}. Use the saved profile or recalibrate.")

    def transform_to_common_frame(self, serial, point):
        self.get_camera_intrinsics(serial)
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Expected a finite camera-space XYZ in metres.")
        if serial == self.serials[0]:
            return p.copy()
        # 열벡터 정의: P_cam1 = R_2to1 @ P_cam2 + t_2to1 (단위 m).
        return self.cam1_from_cam2[:3, :3] @ p + self.cam1_from_cam2[:3, 3]

    def pixel_to_3d(self, serial, uv, depth_m, timestamp):
        intr = self.get_camera_intrinsics(serial)
        result = dict(timestamp=timestamp, valid=False, position_common_m=None)
        u, v = uv
        if (depth_m is None or not np.isfinite(depth_m) or depth_m <= 0
                or not (0 <= u < intr.width and 0 <= v < intr.height)):
            return result
        xy = ideal_pixels(intr, [uv]).reshape(2)
        ray = (xy - [intr.ppx, intr.ppy]) / [intr.fx, intr.fy]
        p = self.transform_to_common_frame(serial, np.r_[ray, 1.] * depth_m)
        return dict(timestamp=timestamp, valid=True, position_common_m=p.tolist())

    def epipolar_errors(self, pixels_by_serial):
        a, b = self.serials
        # F maps ideal camera-2 pixels to epipolar lines in camera 1.
        r, t = self.cam1_from_cam2[:3, :3], self.cam1_from_cam2[:3, 3]
        tx, ty, tz = t
        cross = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        k1, k2 = (camera_matrix(self.intrinsics[s]) for s in (a, b))
        f = np.linalg.inv(k1).T @ cross @ r @ np.linalg.inv(k2)
        p1, p2 = [cv2.convertPointsToHomogeneous(ideal_pixels(self.intrinsics[s], pixels_by_serial[s]))
                  .reshape(-1, 3) for s in (a, b)]
        l1, l2 = p2 @ f.T, p1 @ f
        denominator = np.minimum(np.linalg.norm(l1[:, :2], axis=1), np.linalg.norm(l2[:, :2], axis=1))
        return np.divide(np.abs((p1*l1).sum(axis=1)), denominator,
                         out=np.full(len(p1), np.inf), where=denominator > 1e-12)

    def save(self, path):
        data = dict(version=1, serials=self.serials, rgb_intrinsics={s: asdict(v) for s,v in self.intrinsics.items()},
                    common_rgb_serial=self.serials[0], R_2to1=self.cam1_from_cam2[:3,:3].tolist(),
                    t_2to1_m=self.cam1_from_cam2[:3,3].tolist(),
                    T_cam1_from_cam2=self.cam1_from_cam2.tolist(), metadata=self.metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(data, indent=2, allow_nan=False)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")


def load_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("Unknown stereo calibration format.")
    return StereoCalibration(tuple(data["serials"]),
                             {s: CameraIntrinsics(**v) for s,v in data["rgb_intrinsics"].items()},
                             np.asarray(data["T_cam1_from_cam2"], np.float64), data["metadata"])


def fit_stereo(serials, intrinsics, samples, pattern, square_mm, fps, scales):
    """Fit fixed factory intrinsics, removing high per-pair reprojection outliers."""
    if len(samples) < 6:
        raise ValueError("Capture at least six different board poses before solving.")
    board = board_points(pattern, square_mm)
    views = [[ideal_pixels(intrinsics[s], pair[i]).astype(np.float32) for pair in samples]
             for i, s in enumerate(serials)]
    if any(p.size != 2*len(board) or not np.isfinite(p).all() for view in views for p in view):
        raise ValueError("Invalid or incomplete checkerboard correspondence.")
    left, right = [intrinsics[s] for s in serials]
    selected = list(range(len(samples)))
    while True:
        # OpenCV는 cam1 -> cam2를 반환한다. 출력은 역변환 cam2 -> cam1로 저장한다.
        fit = cv2.stereoCalibrateExtended(
            [board]*len(selected), [views[0][i] for i in selected], [views[1][i] for i in selected],
            camera_matrix(left), np.zeros(5), camera_matrix(right), np.zeros(5),
            (left.width, left.height), None, None, flags=cv2.CALIB_FIX_INTRINSIC)
        rms, r12, t12 = fit[0], fit[5], fit[6]
        errors = np.asarray(fit[-1]).reshape(len(selected), -1).max(axis=1)
        if not np.isfinite(errors).all() or not np.isfinite(rms):
            raise ValueError("Unstable calibration. Check corner order and board poses.")
        if max(errors) <= 2.0:
            break
        if len(selected) <= max(6, int(np.ceil(0.7*len(samples)))):
            raise ValueError("Too many pairs exceed 2 px reprojection error. Check A/B orientation and recapture.")
        selected.pop(int(errors.argmax()))
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = r12.T, (-r12.T @ t12).ravel()
    metadata = dict(date_utc=datetime.now(timezone.utc).isoformat(), inner_corners=list(pattern),
                    square_mm=square_mm, fps=fps, depth_scale_m=scales, reprojection_rms_px=float(rms),
                    accepted_pairs=[i+1 for i in selected], captured_pairs=len(samples),
                    pair_max_view_rms_px=errors.tolist())
    return StereoCalibration(tuple(serials), intrinsics, transform, metadata)

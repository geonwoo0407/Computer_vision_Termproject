"""RGB video input used by the SeatSense vision module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import queue
import threading
import time
from typing import Iterator
import warnings

import cv2
import numpy as np


class InputSourceError(RuntimeError):
    """Raised when a camera or video file cannot be opened."""


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    name: str
    serial_number: str
    firmware_version: str
    usb_type: str
    has_color_sensor: bool
    has_depth_sensor: bool


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    timestamp: float
    camera_id: str
    frame_index: int
    depth_m: np.ndarray | None = None
    intrinsics: CameraIntrinsics | None = None
    serial_number: str | None = None
    color_timestamp_ms: float | None = None
    depth_timestamp_ms: float | None = None
    timestamp_domain: str | None = None
    native_intrinsics: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    received_monotonic: float | None = None


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    """Return connected RealSense devices without starting their streams."""
    try:
        import pyrealsense2 as rs
    except (ImportError, OSError) as exc:
        raise InputSourceError(
            "pyrealsense2 is not available. Check the current environment first."
        ) from exc

    try:
        return _read_realsense_devices(rs)
    except RuntimeError as exc:
        if "timed out" not in str(exc).lower():
            raise InputSourceError(f"RealSense USB device discovery failed: {exc}") from exc

    # A reconnecting UVC device may enumerate before it accepts control queries.
    time.sleep(1)
    try:
        return _read_realsense_devices(rs)
    except RuntimeError as exc:
        raise InputSourceError(
            f"RealSense USB device discovery failed after retry: {exc}. "
            "Check the USB connection and run python main.py --list-cameras."
        ) from exc


def _read_realsense_devices(rs) -> list[RealSenseDeviceInfo]:
    devices = []
    context = rs.context()
    for device in context.query_devices():
        has_color = False
        has_depth = False
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                has_color |= profile.stream_type() == rs.stream.color
                has_depth |= profile.stream_type() == rs.stream.depth

        def read_info(field: object, default: str = "unknown") -> str:
            try:
                return str(device.get_info(field))
            except RuntimeError:
                return default

        devices.append(
            RealSenseDeviceInfo(
                name=read_info(rs.camera_info.name),
                serial_number=str(device.get_info(rs.camera_info.serial_number)),
                firmware_version=read_info(rs.camera_info.firmware_version),
                usb_type=read_info(rs.camera_info.usb_type_descriptor),
                has_color_sensor=has_color,
                has_depth_sensor=has_depth,
            )
        )
    return devices


class VideoSource:
    """Read frames from a webcam index or an RGB video file."""

    def __init__(self, source: int | str, camera_id: str = "camera_1") -> None:
        self.source = source
        self.camera_id = camera_id
        self._capture: cv2.VideoCapture | None = None
        self._is_file = isinstance(source, str)

    def open(self) -> "VideoSource":
        if self._is_file and not Path(str(self.source)).is_file():
            raise InputSourceError(f"Video file not found: {self.source}")

        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            capture.release()
            raise InputSourceError(
                f"Could not open input source {self.source!r}. "
                "Check the camera index, connection, or video path."
            )

        self._capture = capture
        self.frames_received = 0
        return self

    @property
    def frame_size(self) -> tuple[int, int]:
        self._require_open()
        assert self._capture is not None
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    @property
    def fps(self) -> float:
        self._require_open()
        assert self._capture is not None
        return float(self._capture.get(cv2.CAP_PROP_FPS))

    def frames(self) -> Iterator[FramePacket]:
        self._require_open()
        assert self._capture is not None

        frame_index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break

            if self._is_file:
                timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                timestamp = time.time()

            self.frames_received += 1
            yield FramePacket(
                frame=frame,
                timestamp=float(timestamp),
                camera_id=self.camera_id,
                frame_index=frame_index,
                color_timestamp_ms=float(timestamp * 1000.0),
                timestamp_domain=("video_position" if self._is_file else "system_time"),
            )
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _require_open(self) -> None:
        if self._capture is None:
            raise InputSourceError("Input source is not open. Call open() first.")

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


class RealSenseSource:
    """Read color frames, optionally with aligned metric depth."""

    def __init__(
        self,
        serial: str | None = None,
        camera_id: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        include_depth: bool = True,
    ) -> None:
        self.serial = serial
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.include_depth = include_depth
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self.usb_type = "unknown"
        self.frames_received = 0
        self.missing_color_frames = 0
        self.missing_depth_frames = 0
        self.timeouts = 0

    def open(self) -> "RealSenseSource":
        try:
            import pyrealsense2 as rs
        except (ImportError, OSError) as exc:
            raise InputSourceError(
                "pyrealsense2 is not available. Install requirements.txt first."
            ) from exc

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial is not None:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.requested_fps,
        )
        if self.include_depth:
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.requested_fps,
            )

        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            raise InputSourceError(
                f"Could not start RealSense {self.serial or '(auto)'} at "
                f"{self.width}x{self.height}@{self.requested_fps} FPS: {exc}"
            ) from exc

        device = profile.get_device()
        self.serial = device.get_info(rs.camera_info.serial_number)
        self.camera_id = self.camera_id or f"realsense_{self.serial}"
        if self.include_depth:
            self._depth_scale = device.first_depth_sensor().get_depth_scale()
        try:
            self.usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        except RuntimeError:
            self.usb_type = "unknown"
        if self.usb_type.startswith("2"):
            warnings.warn(
                f"RealSense {self.serial} is connected through USB {self.usb_type}; "
                "Frames may time out at the requested settings.",
                RuntimeWarning,
                stacklevel=2,
            )

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.width = color_profile.width()
        self.height = color_profile.height()
        self.requested_fps = color_profile.fps()
        self.frames_received = 0
        self.missing_color_frames = 0
        self.missing_depth_frames = 0
        self.timeouts = 0
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.include_depth else None
        return self

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def fps(self) -> float:
        return float(self.requested_fps)

    @property
    def depth_scale_m(self) -> float:
        if self._pipeline is None or not self.include_depth:
            raise InputSourceError("Depth scale requires an open depth stream.")
        return float(self._depth_scale)

    def frames(self) -> Iterator[FramePacket]:
        if self._pipeline is None:
            raise InputSourceError("RealSense source is not open. Call open() first.")

        frame_index = 0
        while True:
            try:
                frames = self._pipeline.wait_for_frames(5000)
                received_monotonic = time.monotonic()
                if self._align is not None:
                    frames = self._align.process(frames)
            except RuntimeError as exc:
                self.timeouts += 1
                raise InputSourceError(
                    f"Timed out waiting for RealSense {self.serial} frames."
                ) from exc

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame() if self.include_depth else None
            if not color_frame:
                self.missing_color_frames += 1
            if self.include_depth and not depth_frame:
                self.missing_depth_frames += 1
                continue
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            depth_m = None
            depth_timestamp_ms = None
            if depth_frame:
                raw_depth = np.asanyarray(depth_frame.get_data())
                depth_m = raw_depth.astype(np.float32) * self._depth_scale
                depth_timestamp_ms = float(depth_frame.get_timestamp())
            rs_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = CameraIntrinsics(
                width=rs_intrinsics.width,
                height=rs_intrinsics.height,
                fx=float(rs_intrinsics.fx),
                fy=float(rs_intrinsics.fy),
                ppx=float(rs_intrinsics.ppx),
                ppy=float(rs_intrinsics.ppy),
                distortion_model=str(rs_intrinsics.model),
                coefficients=tuple(float(value) for value in rs_intrinsics.coeffs),
            )

            yield FramePacket(
                frame=frame,
                timestamp=float(color_frame.get_timestamp() / 1000.0),
                camera_id=str(self.camera_id),
                frame_index=frame_index,
                depth_m=depth_m,
                intrinsics=intrinsics,
                serial_number=self.serial,
                color_timestamp_ms=float(color_frame.get_timestamp()),
                depth_timestamp_ms=depth_timestamp_ms,
                timestamp_domain=str(color_frame.get_frame_timestamp_domain()),
                native_intrinsics=rs_intrinsics,
                received_monotonic=received_monotonic,
            )
            frame_index += 1

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
                self._align = None

    def __enter__(self) -> "RealSenseSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


class MultiRealSenseSource:
    """Read the latest frame from multiple independent RealSense pipelines."""

    def __init__(
        self,
        serials: list[str],
        camera_ids: list[str] | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        include_depth: bool = True,
    ) -> None:
        if len(serials) < 2:
            raise ValueError("MultiRealSenseSource requires at least two serials")
        if len(set(serials)) != len(serials):
            raise ValueError("RealSense serials must be unique")
        camera_ids = camera_ids or [
            f"camera_{index + 1}" for index in range(len(serials))
        ]
        if len(camera_ids) != len(serials):
            raise ValueError("camera_ids and serials must have the same length")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("camera_ids must be unique")

        self.sources = [
            RealSenseSource(serial, camera_id, width, height, fps, include_depth=include_depth)
            for serial, camera_id in zip(serials, camera_ids)
        ]
        self.width = width
        self.height = height
        self.requested_fps = fps
        self.dropped_frames = {camera_id: 0 for camera_id in camera_ids}
        self._queues = {
            camera_id: queue.Queue(maxsize=1) for camera_id in camera_ids
        }
        self._errors: queue.Queue[tuple[str, Exception]] = queue.Queue()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def fps(self) -> float:
        return float(self.requested_fps)

    def open(self) -> "MultiRealSenseSource":
        opened: list[RealSenseSource] = []
        try:
            for source in self.sources:
                source.open()
                opened.append(source)
        except Exception:
            for source in reversed(opened):
                source.close()
            raise

        first = self.sources[0]
        self.width, self.height = first.frame_size
        self.requested_fps = int(first.fps)
        self._stop_event.clear()
        self._threads = []
        for source in self.sources:
            thread = threading.Thread(
                target=self._capture_loop,
                args=(source,),
                name=f"realsense-{source.camera_id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        return self

    def _capture_loop(self, source: RealSenseSource) -> None:
        assert source.camera_id is not None
        frame_queue = self._queues[source.camera_id]
        try:
            for packet in source.frames():
                if self._stop_event.is_set():
                    break
                try:
                    frame_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.dropped_frames[source.camera_id] += 1
                    frame_queue.put_nowait(packet)
        except Exception as exc:
            if not self._stop_event.is_set():
                self._errors.put((source.camera_id, exc))

    def frames(self) -> Iterator[FramePacket]:
        if not self._threads:
            raise InputSourceError("Multi-camera source is not open. Call open() first.")

        while not self._stop_event.is_set():
            try:
                camera_id, exc = self._errors.get_nowait()
            except queue.Empty:
                pass
            else:
                raise InputSourceError(
                    f"RealSense capture failed for {camera_id}: {exc}"
                ) from exc

            emitted = False
            for frame_queue in self._queues.values():
                try:
                    packet = frame_queue.get_nowait()
                except queue.Empty:
                    continue
                emitted = True
                yield packet

            if not emitted:
                if not any(thread.is_alive() for thread in self._threads):
                    break
                time.sleep(0.001)

    def reset_drop_counts(self) -> None:
        for camera_id in self.dropped_frames:
            self.dropped_frames[camera_id] = 0

    def close(self) -> None:
        self._stop_event.set()
        for source in self.sources:
            source.close()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads = []

    def __enter__(self) -> "MultiRealSenseSource":
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()


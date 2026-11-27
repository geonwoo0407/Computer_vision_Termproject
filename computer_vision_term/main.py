"""SeatSense RGB and RealSense RGB-D command-line runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from camera import (
    InputSourceError,
    MultiRealSenseSource,
    RealSenseSource,
    VideoSource,
    list_realsense_devices,
)
from detector import DetectorDependencyError, VisionDetector
from run_log import open_run_log
from seat_roi import SeatROIManager
from stereo_geometry import load_calibration


def load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as config_file:
            return json.load(config_file)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config: {path} ({exc})") from exc


def parse_source(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SeatSense detection on RGB video or RealSense RGB-D."
    )
    parser.add_argument("--config", default="config.json", help="JSON config path")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List RealSense devices without loading YOLO",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--camera-only",
        action="store_true",
        help="Test RealSense RGB-D frames without loading YOLO",
    )
    modes.add_argument(
        "--chair-only",
        action="store_true",
        help="Detect the COCO chair class in full RGB frames",
    )
    modes.add_argument("--single-seat", action="store_true",
                       help="Observe one fixed seat using YOLO, an empty RGB reference and timed votes")
    modes.add_argument("--six-seats", action="store_true",
                       help="Observe two rows of three seats in a single fixed RGB view")
    parser.add_argument("--grid-layout", type=Path, help="Six-seat ROI file (default: output/seat_grid_3x2.json)")
    parser.add_argument("--app-url", help="SeatSense base URL for confirmed six-seat observations")
    parser.add_argument("--app-seats", nargs=6, type=int, metavar="SEAT",
                        help="Six app seat numbers in camera slot order (far row first, then near row)")
    parser.add_argument("--desk-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--chair-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--confirm-seconds", type=float, help="Seat vote window (default 5s, minimum 5s)")
    parser.add_argument(
        "--input-type",
        choices=("video", "realsense"),
        help="Input backend (default comes from config.json)",
    )
    parser.add_argument("--source", help="Webcam index (for example 0) or video path")
    serial_group = parser.add_mutually_exclusive_group()
    serial_group.add_argument("--serial", help="RealSense serial number")
    serial_group.add_argument(
        "--serials",
        nargs="+",
        help="Two or more RealSense serials for multi-camera input",
    )
    parser.add_argument("--camera-id", help="Camera identifier included in each result")
    parser.add_argument("--stereo", help="Stereo calibration JSON for common camera-1 XYZ in RGB-D output")
    parser.add_argument("--model", help="Ultralytics model name or local weights path")
    parser.add_argument("--confidence", type=float, help="Detection confidence threshold")
    parser.add_argument("--device", help="Inference device: cpu, 0, 1, ...")
    parser.add_argument("--image-size", type=int, help="YOLO inference image size")
    parser.add_argument("--width", type=int, help="RealSense color/depth width")
    parser.add_argument("--height", type=int, help="RealSense color/depth height")
    parser.add_argument("--fps", type=int, help="RealSense stream FPS")
    parser.add_argument(
        "--seat-rois",
        help="Optional seat ROI JSON file",
    )
    parser.add_argument(
        "--jsonl",
        help="Set the frame log path (default: a new file under output/logs/)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the OpenCV preview window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after this many frames (useful for a smoke test)",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Ignore this many initial frames in --camera-only statistics",
    )
    return parser


def print_realsense_devices() -> int:
    devices = list_realsense_devices()
    print(f"Found {len(devices)} RealSense device(s).")
    for index, device in enumerate(devices, start=1):
        print(f"\n[{index}] {device.name}")
        print(f"  serial: {device.serial_number}")
        print(f"  firmware: {device.firmware_version}")
        print(f"  usb: {device.usb_type}")
        print(f"  color: {'yes' if device.has_color_sensor else 'no'}")
        print(f"  depth: {'yes' if device.has_depth_sensor else 'no'}")
    return 0


def show_camera_window(
    title_prefix: str,
    camera_id: str,
    frame: np.ndarray,
    window_indices: dict[str, int],
) -> None:
    """Show one persistent, side-by-side window per camera."""
    window_name = f"{title_prefix} - {camera_id} - press q to quit"
    if camera_id not in window_indices:
        window_index = len(window_indices)
        window_indices[camera_id] = window_index
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        height, width = frame.shape[:2]
        cv2.moveWindow(window_name, 30 + window_index * (width + 30), 60)
    cv2.imshow(window_name, frame)


def run_camera_diagnostics(
    source: RealSenseSource | MultiRealSenseSource,
    max_frames: int | None,
    no_display: bool,
    min_depth_m: float,
    max_depth_m: float,
    warmup_frames: int,
    log=None,
) -> int:
    states: dict[str, dict[str, Any]] = {}
    sources = source.sources if isinstance(source, MultiRealSenseSource) else [source]
    for camera_source in sources:
        assert camera_source.camera_id is not None
        states[camera_source.camera_id] = {
            "source": camera_source,
            "started": None,
            "captured": 0,
            "processed": 0,
            "valid_ratios": [],
            "timestamp_differences": [],
            "non_monotonic_timestamps": 0,
            "previous_timestamp": None,
        }
    window_indices: dict[str, int] = {}

    for packet in source.frames():
        state = states[packet.camera_id]
        if max_frames is not None and state["processed"] >= max_frames:
            if all(item["processed"] >= max_frames for item in states.values()):
                break
            continue
        state["captured"] += 1
        if state["captured"] <= warmup_frames:
            continue
        if state["processed"] == 0:
            state["started"] = time.perf_counter()
        state["processed"] += 1
        valid_ratio = 0.0
        if packet.depth_m is not None:
            valid = (
                np.isfinite(packet.depth_m)
                & (packet.depth_m >= min_depth_m)
                & (packet.depth_m <= max_depth_m)
            )
            valid_ratio = float(np.count_nonzero(valid) / valid.size)
            state["valid_ratios"].append(valid_ratio)
        if log is not None:
            print(json.dumps({"mode": "camera_only", "camera_id": packet.camera_id,
                              "serial_number": packet.serial_number, "frame_id": packet.frame_index,
                              "timestamp": packet.timestamp, "color_timestamp_ms": packet.color_timestamp_ms,
                              "depth_timestamp_ms": packet.depth_timestamp_ms,
                              "valid_depth_ratio": valid_ratio if packet.depth_m is not None else None}),
                  file=log, flush=True)

        if (
            packet.color_timestamp_ms is not None
            and packet.depth_timestamp_ms is not None
        ):
            state["timestamp_differences"].append(
                abs(packet.color_timestamp_ms - packet.depth_timestamp_ms)
            )
        if (
            state["previous_timestamp"] is not None
            and packet.timestamp <= state["previous_timestamp"]
        ):
            state["non_monotonic_timestamps"] += 1
        state["previous_timestamp"] = packet.timestamp

        if not no_display:
            preview = packet.frame.copy()
            status = (
                f"{packet.camera_id} | frame {packet.frame_index} | "
                f"valid depth {valid_ratio:.1%}"
            )
            cv2.putText(
                preview,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            show_camera_window(
                "SeatSense Camera Test",
                packet.camera_id,
                preview,
                window_indices,
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if max_frames is not None and all(
            item["processed"] >= max_frames for item in states.values()
        ):
            break

    ended = time.perf_counter()
    print("\nCamera diagnostic summary")
    for camera_id, state in states.items():
        camera_source = state["source"]
        started = state["started"] or ended
        elapsed = max(ended - started, 1e-9)
        valid_ratios = state["valid_ratios"]
        timestamp_differences = state["timestamp_differences"]
        print(f"\n[{camera_id}]")
        print(f"  serial: {camera_source.serial}")
        print(f"  usb: {camera_source.usb_type}")
        print(f"  warmup_frames: {min(state['captured'], warmup_frames)}")
        print(f"  measured_frames: {state['processed']}")
        print(f"  total_frames_received: {state['captured']}")
        print(f"  missing_color_frames: {camera_source.missing_color_frames}")
        print(f"  missing_depth_frames: {camera_source.missing_depth_frames}")
        print(f"  timeouts: {camera_source.timeouts}")
        print(f"  measured_camera_fps: {state['processed'] / elapsed:.2f}")
        print(
            "  average_valid_depth_ratio: "
            f"{(sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0):.4f}"
        )
        print(
            "  average_color_depth_timestamp_difference_ms: "
            f"{(sum(timestamp_differences) / len(timestamp_differences) if timestamp_differences else 0.0):.3f}"
        )
        print(
            "  non_monotonic_timestamps: "
            f"{state['non_monotonic_timestamps']}"
        )
        if isinstance(source, MultiRealSenseSource):
            print(f"  queue_dropped_frames: {source.dropped_frames[camera_id]}")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.list_cameras:
        return print_realsense_devices()

    config = load_config(args.config)
    seat_mode = args.single_seat or args.six_seats
    if (args.app_url or args.app_seats) and not (args.six_seats and args.app_url):
        raise ValueError("--app-seats and --app-url require --six-seats with an app URL.")
    if seat_mode:
        if args.serials or args.stereo or args.seat_rois:
            raise ValueError("Seat preview takes one RGB source and its own desk/chair ROIs.")
        if args.confirm_seconds is not None and not (np.isfinite(args.confirm_seconds) and args.confirm_seconds >= 5):
            raise ValueError("--confirm-seconds must be finite and at least 5.")
    elif args.confirm_seconds is not None:
        raise ValueError("--confirm-seconds requires --single-seat or --six-seats.")
    if args.single_seat:
        from single_seat import draw_status, prepare_single_seat
        if bool(args.desk_roi) != bool(args.chair_roi) or (args.no_display and not args.desk_roi):
            raise ValueError("Provide both --desk-roi and --chair-roi for preset or headless operation.")
    elif args.desk_roi or args.chair_roi:
        raise ValueError("--desk-roi and --chair-roi require --single-seat.")
    if args.six_seats:
        from seat_grid import DEFAULT_LAYOUT, draw_grid, prepare_grid
        layout = args.grid_layout or DEFAULT_LAYOUT
        if args.jsonl and layout.resolve() == Path(args.jsonl).resolve():
            raise ValueError("--jsonl and --grid-layout must be different files.")
    elif args.grid_layout:
        raise ValueError("--grid-layout requires --six-seats.")
    input_type = args.input_type or config.get("input_type", "video")
    source = parse_source(args.source or str(config.get("source", "0")))
    serial = args.serial or (None if seat_mode else config.get("realsense_serial"))
    camera_id = args.camera_id or config.get("camera_id")
    stereo = load_calibration(args.stereo) if args.stereo else None
    if stereo and (input_type != "realsense" or args.chair_only or args.camera_only):
        raise ValueError("--stereo requires RealSense RGB-D object detection.")
    if stereo:
        for requested in args.serials or ([serial] if serial else []):
            stereo.get_camera_intrinsics(requested)

    if args.camera_only and input_type != "realsense":
        raise ValueError("--camera-only requires --input-type realsense")

    if input_type == "realsense":
        width = args.width or int(config.get("realsense_width", 640))
        height = args.height or int(config.get("realsense_height", 480))
        fps = args.fps or int(config.get("realsense_fps", 30))
        if args.serials:
            if len(args.serials) < 2:
                raise ValueError("--serials requires at least two serial numbers")
            input_source = MultiRealSenseSource(
                serials=args.serials,
                width=width,
                height=height,
                fps=fps,
                include_depth=not args.chair_only,
            )
            input_label = f"RealSense serials={','.join(args.serials)}"
        else:
            input_source = RealSenseSource(
                serial=serial,
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps,
                include_depth=not (args.chair_only or seat_mode),
            )
            input_label = f"RealSense serial={serial or 'auto'}"
    else:
        input_source = VideoSource(source, camera_id or "camera_1")
        input_label = repr(source)

    seat_roi_path = args.seat_rois or config.get("seat_roi_file")
    seat_rois = (
        SeatROIManager(
            seat_roi_path,
            min_overlap=float(config.get("seat_min_overlap", 0.05)),
        )
        if seat_roi_path and not seat_mode
        else None
    )

    output_file = None

    processed_frames = 0
    smoothed_pipeline_fps = 0.0
    previous_time = time.perf_counter()
    previous_camera_times: dict[str, float] = {}
    smoothed_camera_fps: dict[str, float] = {}
    window_indices: dict[str, int] = {}
    publisher = None
    try:
        if args.app_url:
            from seat_app import GridPublisher, SeatApp
            app_seats = args.app_seats or config.get("six_seat_app_ids", [])
            publisher = GridPublisher(SeatApp(args.app_url), app_seats)
            print(f"SeatSense {args.app_url}: camera slots 1..6 -> {app_seats}", flush=True)
        elif args.six_seats:
            print("SeatSense app offline: preview only. To sync, set --app-url.", flush=True)
        with input_source as video:
            mode = next((name for name in ("six_seats", "single_seat", "chair_only", "camera_only")
                         if getattr(args, name)), "detections")
            output_file = open_run_log(mode, args.jsonl)
            print(
                f"Input opened: {input_label}, size={video.frame_size}, "
                f"reported_fps={video.fps:.2f}"
            )
            if args.camera_only:
                depth_config = config.get("depth", {})
                return run_camera_diagnostics(
                    video,
                    args.max_frames,
                    args.no_display,
                    float(depth_config.get("min_depth_m", 0.1)),
                    float(depth_config.get("max_depth_m", 10.0)),
                    args.warmup_frames,
                    log=output_file,
                )

            classes = config.get("target_classes")
            if args.chair_only:
                classes = ["chair"]
            if seat_mode:
                classes = None
            detector = VisionDetector(
                model_path=args.model or config.get("model", "yolo11n.pt"),
                confidence=(
                    args.confidence
                    if args.confidence is not None
                    else float(config.get("confidence", 0.35))
                ),
                image_size=args.image_size or int(config.get("image_size", 640)),
                device=args.device or config.get("device", "cpu"),
                target_classes=classes,
            )
            if isinstance(video, MultiRealSenseSource):
                video.reset_drop_counts()
            checked_profiles = set()
            packets = iter(video.frames())
            seat = None
            if args.single_seat:
                seat = prepare_single_seat(packets, detector, args.desk_roi, args.chair_roi,
                                           args.no_display, args.confirm_seconds or 5.0)
                seat_key, draw_seats = "single_seat", draw_status
            elif args.six_seats:
                seat = prepare_grid(packets, detector, layout, args.no_display, args.confirm_seconds or 5.0)
                seat_key, draw_seats = "seat_grid", draw_grid
            if seat_mode and seat is None:
                return 0
            for packet in packets:
                captured = packet.received_monotonic if packet.received_monotonic is not None else time.monotonic()
                if stereo and packet.serial_number not in checked_profiles:
                    stereo.check_profile(packet.serial_number, packet.intrinsics, video.fps)
                    checked_profiles.add(packet.serial_number)
                result = detector.process_packet(packet)
                if seat is not None:
                    result[seat_key] = seat.observe(packet.frame, result["detections"], packet.timestamp)
                    if publisher is not None:
                        publisher.update(result[seat_key], captured)
                if stereo:
                    result["common_frame"] = f"{stereo.serials[0]}_color_optical_frame"
                    for obj in result["detections"]:
                        local = obj["position_3d_m"]
                        obj["position_common_m"] = (stereo.transform_to_common_frame(packet.serial_number, local).tolist()
                                                    if local is not None and np.isfinite(local).all() else None)
                if seat_rois is not None:
                    result["seat_rois_calibrated"] = seat_rois.calibrated
                    result["seat_observations"] = seat_rois.add_candidates(
                        result["detections"], video.frame_size
                    )
                else:
                    result["seat_rois_calibrated"] = False
                    result["seat_observations"] = []

                now = time.perf_counter()
                instant_pipeline_fps = 1.0 / max(now - previous_time, 1e-9)
                smoothed_pipeline_fps = (
                    instant_pipeline_fps
                    if smoothed_pipeline_fps == 0.0
                    else 0.9 * smoothed_pipeline_fps + 0.1 * instant_pipeline_fps
                )
                previous_time = now

                previous_camera_time = previous_camera_times.get(packet.camera_id)
                if previous_camera_time is None:
                    camera_fps = 0.0
                else:
                    instant_camera_fps = 1.0 / max(
                        now - previous_camera_time,
                        1e-9,
                    )
                    previous_smoothed = smoothed_camera_fps.get(packet.camera_id, 0.0)
                    camera_fps = (
                        instant_camera_fps
                        if previous_smoothed == 0.0
                        else 0.9 * previous_smoothed + 0.1 * instant_camera_fps
                    )
                previous_camera_times[packet.camera_id] = now
                smoothed_camera_fps[packet.camera_id] = camera_fps
                result["fps"] = round(camera_fps, 2)
                result["pipeline_fps"] = round(smoothed_pipeline_fps, 2)
                processed_frames += 1

                print(json.dumps(result, ensure_ascii=False), file=output_file, flush=True)

                if not args.no_display:
                    preview = detector.draw_detections(
                        packet.frame, result, camera_fps
                    )
                    if seat_rois is not None:
                        preview = seat_rois.draw(preview)
                    if seat is not None:
                        preview = draw_seats(preview, result[seat_key])
                    show_camera_window(
                        "SeatSense Vision",
                        packet.camera_id,
                        preview,
                        window_indices,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames is not None and processed_frames >= args.max_frames:
                    break
            if isinstance(video, MultiRealSenseSource):
                print("Multi-camera queue drops:")
                for drop_camera_id, drop_count in video.dropped_frames.items():
                    print(f"  {drop_camera_id}: {drop_count}")
    finally:
        if publisher is not None:
            publisher.close()
        if output_file is not None:
            output_file.close()
        cv2.destroyAllWindows()

    print(f"Finished normally. Processed {processed_frames} frame(s).")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must not be negative")
    for name in ("width", "height", "fps", "image_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    try:
        return run(args)
    except (DetectorDependencyError, InputSourceError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


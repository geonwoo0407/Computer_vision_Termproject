"""Calibrate two fixed RGB cameras, fuse observations and feed the SeatSense app."""

import argparse
import base64
from contextlib import closing
from dataclasses import asdict
import http.client
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from camera import InputSourceError, MultiRealSenseSource, RealSenseSource
from desk import REFERENCE_SAMPLES, sampled_frames
from detector import DetectorDependencyError, VisionDetector
from main import load_config, show_camera_window
from seat_fusion import MAX_SAMPLE_GAP, PATCH_CORNERS, SeatFusion, SeatView, observe_view
from stereo_geometry import load_calibration


def select_table(window, frozen, seat):
    points = []

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.setMouseCallback(window, click)
    while True:
        preview = frozen.copy()
        cv2.putText(preview, f"Seat {seat}: click A B C D | Enter: accept | R: redo | Q: quit",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 230, 230), 2)
        for label, point in zip("ABCD", points):
            cv2.putText(preview, label, point, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        if len(points) >= 2:
            cv2.polylines(preview, [np.int32(points)], len(points) == 4, (30, 210, 240), 2)
        cv2.imshow(window, preview)
        key = cv2.waitKey(25) & 255
        if key in (13, 10) and len(points) == 4:
            return points
        if key == ord("r"):
            points.clear()
        if key in (ord("q"), 27):
            raise KeyboardInterrupt


def calibrate(args):
    if args.file.exists():
        raise ValueError("Calibration file already exists. Choose a new filename.")
    if len(set(args.serials)) != 2 or len(set(args.seats)) != len(args.seats):
        raise ValueError("Camera serials and seat numbers must be distinct.")
    stereo = load_calibration(args.stereo) if args.stereo else None
    if stereo and set(stereo.serials) != set(args.serials):
        raise ValueError("The stereo calibration belongs to a different camera pair.")
    config = load_config(args.config)
    width, height, fps = (config["realsense_"+k] for k in ("width","height","fps"))
    views, images = [], {}
    for index, serial in enumerate(args.serials):
        window = f"SeatSense setup - {serial}"
        with RealSenseSource(serial=serial, width=width, height=height, fps=fps, include_depth=False) as source:
            stream = source.frames()
            for packet in stream:
                preview = packet.frame.copy()
                cv2.putText(preview, "Clear the desk and chair | F: capture | Q: quit",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 240), 2)
                cv2.imshow(window, preview)
                key = cv2.waitKey(1) & 255
                if key in (ord("q"), 27):
                    raise KeyboardInterrupt
                if key == ord("f"):
                    break
            if stereo:
                stereo.check_profile(serial, packet.intrinsics, source.fps)
            samples = [p.frame.copy() for p in islice(sampled_frames(stream, packet.frame.shape), REFERENCE_SAMPLES)]
            if len(samples) != REFERENCE_SAMPLES:
                raise ValueError("Not enough empty reference frames.")
            reference = np.median(samples, axis=0).astype(np.uint8)
        seats = {}
        for seat in args.seats:
            while True:
                table = select_table(window, reference, seat)
                if not stereo or index == 0:
                    break
                first = views[0]["seats"][str(seat)]["table"]
                errors = stereo.epipolar_errors({args.serials[0]: first, serial: table})
                print(f"Seat {seat}, epipolar errors A/B/C/D (px): {errors.round(2)}", flush=True)
                if np.max(errors) <= 5.0:
                    break
                print("Correspondence exceeds 5 px. Reselect the same physical A/B/C/D in camera 2.", flush=True)
            guide = reference.copy()
            cv2.polylines(guide, [np.int32(table)], True, (30, 210, 240), 2)
            cv2.putText(guide, f"Seat {seat}: drag seated-person / chair region; Enter",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 220, 240), 2)
            chair = cv2.selectROI(window, guide, fromCenter=False, showCrosshair=True)
            if not all(chair[2:]):
                raise ValueError("Seat setup cancelled.")
            SeatView(reference, table, chair)
            seats[str(seat)] = {"table": table, "chair": list(map(int, chair))}
            if stereo and index == 1:
                seats[str(seat)]["epipolar_errors_px"] = errors.tolist()
        views.append({"serial": serial, "seats": seats})
        images[f"reference_{index}"] = reference
        cv2.destroyWindow(window)
    args.file.parent.mkdir(parents=True, exist_ok=True)
    with args.file.open("xb") as output:
        np.savez_compressed(output, views=json.dumps(views), **images)
    print(f"Saved calibration for {args.seats} to {args.file}")


def read_views(path):
    with np.load(path, allow_pickle=False) as archive:
        layout = json.loads(str(archive["views"]))
        views = {}
        for index, camera in enumerate(layout):
            reference = archive[f"reference_{index}"]
            views[camera["serial"]] = {
                int(seat): SeatView(reference, spec["table"], spec["chair"])
                for seat, spec in camera["seats"].items()
            }
    if len(views) != 2:
        raise ValueError("Expected two distinct calibrated cameras.")
    left, right = views.values()
    if not left or left.keys() != right.keys() or any(not 1 <= seat <= 288 for seat in left):
        raise ValueError("Both cameras must cover the same selected SeatSense seats (1..288).")
    return views


class SeatApp:
    def __init__(self, url):
        address = urlsplit(url)
        if (not address.hostname or address.username or address.password or address.query
                or address.fragment or address.path not in ("", "/")):
            raise ValueError("--app-url must be the SeatSense server's base URL.")
        if address.scheme == "https":
            self.connection = http.client.HTTPSConnection
        elif address.scheme == "http" and address.hostname in ("localhost", "127.0.0.1", "::1"):
            self.connection = http.client.HTTPConnection
        else:
            raise ValueError("Use HTTPS for a remote SeatSense server or HTTP on localhost.")
        self.host, self.port = address.hostname, address.port
        password = os.getenv("SEAT_ADMIN_PASSWORD")
        if not password:
            raise ValueError("Set SEAT_ADMIN_PASSWORD to the app's configured admin password.")
        token = base64.b64encode(f"admin:{password}".encode()).decode()
        self.headers = {"Content-Type": "application/json", "Authorization": "Basic " + token}
        self.delivered = {}
        self.retry_at = 0.0

    def send(self, seat, state, captured):
        now = time.monotonic()
        if self.delivered.get(seat) == state or now < self.retry_at or now-captured > MAX_SAMPLE_GAP:
            return
        payload = dict(action="detect", source="camera", seat_id=seat,
                       occupied=state[0], belongings=state[1])
        try:
            with closing(self.connection(self.host, self.port, timeout=2)) as client:
                client.request("POST", "/api/action", json.dumps(payload), self.headers)
                response = client.getresponse()
                status = response.status
                response.read()
            if status == 403:
                raise ValueError("SeatSense rejected admin authentication. Check SEAT_ADMIN_PASSWORD.")
            if status == 409:
                raise ValueError("In SeatSense /demo select 카메라 API 모드, then start again.")
            if 400 <= status < 500:
                raise ValueError(f"SeatSense rejected seat {seat} with HTTP {status}.")
            if status != 200:
                raise OSError(f"SeatSense HTTP {status}")
        except (OSError, http.client.HTTPException) as exc:
            self.delivered.pop(seat, None)
            self.retry_at = time.monotonic() + 2
            print(f"Camera update pending ({exc}); retrying fresh observations.", file=sys.stderr)
            return
        self.delivered[seat] = state
        print(f"SeatSense updated seat {seat}", flush=True)


def report(results, last, app, captured):
    for seat, (status, state) in sorted(results.items()):
        if last.get(seat) != (status, state):
            fields = "" if state is None else f" person={state[0]} belongings={state[1]}"
            print(f"seat {seat}: {status}{fields}", flush=True)
            last[seat] = (status, state)
        if app and status == "stable":
            app.send(seat, state, captured)


def record_observation(log, serial, stamp, observations, results):
    if log is not None:
        row = {"serial": serial, "capture_monotonic_s": stamp,
               "observations": {seat: asdict(obs) for seat, obs in observations.items()},
               "fusion": {seat: {"status": status, "occupied": state[0] if state else None,
                                 "belongings": state[1] if state else None}
                          for seat, (status, state) in results.items()}}
        print(json.dumps(row), file=log, flush=True)


def live(args, app, log):
    views = read_views(args.file)
    serials = list(views)
    fusion = SeatFusion(serials)
    config = load_config(args.config)
    detector = VisionDetector(model_path=config["model"], confidence=config["confidence"],
                              image_size=config["image_size"], device=config["device"],
                              target_classes=config["target_classes"])
    windows, last = {}, {}
    with MultiRealSenseSource(serials, width=config["realsense_width"], height=config["realsense_height"],
                              fps=config["realsense_fps"], include_depth=False) as cameras:
        for packet in cameras.frames():
            serial, captured = packet.serial_number, packet.received_monotonic
            detected = detector.process_frame(packet.frame, packet.camera_id, packet.timestamp)
            seats = views[serial]
            observations = observe_view(packet.frame, seats, detected["detections"])
            results = fusion.update(serial, captured, observations, time.monotonic())
            record_observation(log, serial, captured, observations, results)
            report(results, last, app, captured)
            if not args.no_display:
                preview = detector.draw_detections(packet.frame, detected)
                for seat, view in seats.items():
                    polygon = view.corners.astype(np.int32)
                    cv2.polylines(preview, [polygon], True, (0, 210, 240), 2)
                    x, y, w, h = view.chair_region
                    cv2.rectangle(preview, (x, y), (x+w, y+h), (200, 160, 30), 1)
                    cv2.putText(preview, f"{seat} {observations[seat].table}", tuple(polygon[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 240), 2)
                show_camera_window("SeatSense two views", packet.camera_id, preview, windows)
                if cv2.waitKey(1) & 255 in (27, ord("q")):
                    break


def demo(args, app, log):
    """Exercise the real image comparison, fusion and optional API using two synthetic views."""
    serials = ["demo-front", "demo-rear"]
    fusion = SeatFusion(serials)
    reference = np.full((150, 240, 3), 130, np.uint8)
    corners = [[[15, 62], [220, 51], [207, 129], [28, 133]],
               [[218, 123], [25, 134], [37, 58], [202, 54]]]
    views = [SeatView(reference, quad, (70, 0, 100, 50)) for quad in corners]
    spot = np.float32([[[71, 42], [90, 42], [90, 56], [71, 56]]])
    polygons = [cv2.perspectiveTransform(spot, cv2.getPerspectiveTransform(PATCH_CORNERS, view.corners))
                .round().astype(np.int32)[0] for view in views]
    phases = [("빈자리", False, (False, False), 25),
              ("뒤 카메라에만 사람", True, (False, False), 13),
              ("앞 상판에만 변화", False, (True, False), 13),
              ("두 상판 같은 위치에 소지품", False, (True, True), 16),
              ("두 시점 모두 비워짐", False, (False, False), 25)]
    last, step = {}, 0
    for name, person, objects, samples in phases:
        print(f"[모의 영상] {name}", flush=True)
        for _ in range(samples):
            stamp = time.monotonic() if app else step * 0.25
            for i, serial in enumerate(serials):
                frame = reference.copy()
                if objects[i]:
                    cv2.fillConvexPoly(frame, polygons[i], (20, 40, 190))
                detections = []
                if person and i == 1:
                    detections.append({"class_name": "person", "bbox": [85, 2, 145, 46]})
                observations = observe_view(frame, {args.seat: views[i]}, detections)
                results = fusion.update(serial, stamp, observations, stamp)
                record_observation(log, serial, stamp, observations, results)
                report(results, last, app, stamp)
            step += 1
            if app:
                time.sleep(0.25)
    if app and app.delivered.get(args.seat) != (False, False):
        raise ValueError("Final demo observation did not reach SeatSense.")


def seat_number(text):
    value = int(text)
    if not 1 <= value <= 288:
        raise argparse.ArgumentTypeError("Seat number must be in 1..288.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    setup = modes.add_parser("calibrate", help="Capture empty references and select matching desk corners")
    setup.add_argument("file", type=Path)
    setup.add_argument("--serials", nargs=2, required=True)
    setup.add_argument("--seats", nargs="+", type=seat_number, required=True)
    setup.add_argument("--stereo", type=Path, help="Check selected A/B/C/D points against saved epipolar geometry")
    setup.add_argument("--config", default="config.json")
    capture = modes.add_parser("live", help="Start the calibrated two-camera RGB pipeline")
    capture.add_argument("file", type=Path)
    capture.add_argument("--config", default="config.json")
    capture.add_argument("--no-display", action="store_true")
    simulation = modes.add_parser("demo", help="Run a synthetic two-view sequence without hardware")
    simulation.add_argument("--seat", type=seat_number, default=1)
    for command in (capture, simulation):
        command.add_argument("--app-url", help="SeatSense base URL; omit to print observations locally")
        command.add_argument("--jsonl", type=Path, help="Write per-camera evidence and paired decisions to a new file")
    args = parser.parse_args()
    log = None
    try:
        if args.mode == "calibrate":
            calibrate(args)
        else:
            app = SeatApp(args.app_url) if args.app_url else None
            if args.jsonl:
                args.jsonl.parent.mkdir(parents=True, exist_ok=True)
                log = args.jsonl.open("x", encoding="utf-8")
            if args.mode == "demo":
                demo(args, app, log)
            else:
                live(args, app, log)
    except (DetectorDependencyError, InputSourceError, OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if log is not None:
            log.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

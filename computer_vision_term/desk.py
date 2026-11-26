"""Record RGB tabletop crops and detect changes against an empty reference."""

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import cv2
import numpy as np

from camera import FramePacket, InputSourceError, RealSenseSource


class DeskChangeDetector:
    """Observations for one tabletop ROI in a single fixed camera view."""

    def __init__(self, color):
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("Reference must be an 8-bit BGR image.")
        self.color = cv2.GaussianBlur(color, (3, 3), 0)
        self.chroma = self.chromaticity(self.color)
        self.streak = np.zeros(color.shape[:2], np.uint8)
        self.min_area = max(12, int(np.ceil(self.streak.size * 0.01)))

    @staticmethod
    def chromaticity(bgr):
        # Ratios are less sensitive to a neutral shadow than raw brightness.
        bgr = bgr.astype(np.float32)
        return bgr / np.maximum(bgr.sum(axis=2, keepdims=True), 1.0)

    def regions(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return [row[:4].tolist() for row in stats[1:] if row[4] >= self.min_area]

    def observe(self, color):
        if color.shape != self.color.shape or color.dtype != np.uint8:
            raise ValueError("Expected an 8-bit BGR frame matching the reference.")
        bgr = cv2.GaussianBlur(color, (3, 3), 0)
        rgb_change = cv2.absdiff(self.color, bgr).max(axis=2) >= 25
        color_change = np.abs(self.chromaticity(bgr) - self.chroma).max(axis=2) > 0.07
        evidence = rgb_change & color_change
        self.streak = np.where(evidence, np.minimum(self.streak + 1, 3), 0).astype(np.uint8)
        boxes = self.regions(self.streak >= 3)
        if boxes:
            state = "object_candidate"
        elif self.regions(rgb_change):
            state = "uncertain"
        else:
            state = "no_change"
        return state, boxes


def save_npz(path, **arrays):
    # Only complete files appear to replay, even if recording is interrupted.
    partial = path.with_suffix(".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    partial.replace(path)


DESK_WINDOW = "SeatSense desk"
QUIT_KEYS = (ord("q"), ord("Q"), 27)
REFERENCE_SAMPLES = 15


def annotated_frame(color, state, boxes):
    overlay = color.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (30, 180, 255), 2)
    for thickness, ink in ((3, (0, 0, 0)), (1, (70, 240, 255))):
        cv2.putText(overlay, state, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, ink, thickness, cv2.LINE_AA)
    return overlay


def preview(color, state, boxes, delay=1):
    cv2.imshow(DESK_WINDOW, annotated_frame(color, state, boxes))
    return cv2.waitKey(delay) & 0xFF


def sampled_frames(packets, shape):
    last_sample = None
    for packet in packets:
        # Sample by the color timestamp, at roughly 5 Hz, for bounded disk use.
        ts = packet.color_timestamp_ms
        if ts is None:
            raise ValueError("A color timestamp is required.")
        if last_sample is not None and ts >= last_sample and ts - last_sample < 199:
            continue
        last_sample = ts
        if packet.frame.shape != shape:
            raise ValueError("Color frame dimensions changed during recording.")
        yield packet


def prepare_recording(packets: Iterator[FramePacket], preset_roi, no_display):
    """Return an in-memory reference after the GUI operator starts recording."""
    for view in packets:
        roi = preset_roi
        if not no_display:
            key = preview(view.frame, "LIVE   F: freeze   Q: quit", [roi] if roi else [])
            if key in QUIT_KEYS:
                return None
            if key not in (ord("f"), ord("F")):
                continue
            if roi is None:
                frozen = annotated_frame(view.frame, "FROZEN   Drag ROI   Enter: accept   c: back", [])
                roi = cv2.selectROI(DESK_WINDOW, frozen, fromCenter=False, showCrosshair=True)
                if min(roi[2:]) < 4:
                    continue

        x, y, w, h = roi
        height, width = view.frame.shape[:2]
        if min(x, y) < 0 or min(w, h) < 4 or x+w > width or y+h > height:
            raise ValueError(f"ROI must fit inside {width}x{height}, minimum size 4x4.")
        print("Keep the desk empty for the reference (~3 s).", flush=True)
        colors = []
        for packet in islice(sampled_frames(packets, view.frame.shape), REFERENCE_SAMPLES):
            colors.append(packet.frame[y:y+h, x:x+w].copy())
            if not no_display:
                key = preview(packet.frame, f"EMPTY DESK   {len(colors)}/{REFERENCE_SAMPLES}   Q: quit", [roi])
                if key in QUIT_KEYS:
                    return None
        if len(colors) < REFERENCE_SAMPLES:
            raise ValueError("Input ended before the empty-desk reference was ready.")
        bg = np.median(np.stack(colors), axis=0).astype(np.uint8)
        if no_display:
            return view, roi, bg

        print("Reference ready. R: start recording, B: reframe, Q: quit.", flush=True)
        for ready in packets:
            key = preview(ready.frame, "READY   R: record   B: reframe   Q: quit", [roi])
            if key in (ord("r"), ord("R")):
                return view, roi, bg
            if key in QUIT_KEYS:
                return None
            if key in (ord("b"), ord("B")):
                break
    raise ValueError("RGB input ended during recording setup.")


def record_frames(packets: Iterable[FramePacket], directory: Path, roi, no_display,
                  max_frames: int):
    packets = iter(packets)
    prepared = prepare_recording(packets, roi, no_display)
    if prepared is None:
        print("Recording cancelled.")
        return
    first, roi, bg = prepared
    x, y, w, h = roi
    height, width = first.frame.shape[:2]
    metadata = {
        "version": 2, "serial_number": first.serial_number, "camera_id": first.camera_id,
        "roi_xywh": list(map(int, roi)), "frame_size_wh": [width, height],
        "color_intrinsics_full_frame": asdict(first.intrinsics) if first.intrinsics else None,
        "timestamp_domain": first.timestamp_domain,
        "reference_samples": REFERENCE_SAMPLES,
    }
    detector = DeskChangeDetector(bg)
    directory.mkdir(parents=True, exist_ok=False)
    save_npz(directory / "reference.npz", color_bgr=bg, metadata=json.dumps(metadata))
    print("RECORDING - place and remove the object now.", flush=True)
    count = 0
    previous_state = None
    for packet in sampled_frames(packets, first.frame.shape):
        color = packet.frame[y:y+h, x:x+w]
        save_npz(directory / f"frame_{count:06d}.npz", color_bgr=color,
                 frame_id=packet.frame_index, color_timestamp_ms=packet.color_timestamp_ms)
        count += 1
        state, boxes = detector.observe(color)
        if state != previous_state:
            print(f"frame {packet.frame_index}: {state}", flush=True)
            previous_state = state
        if not no_display and preview(color, f"REC   {state}   Q: stop", boxes) in QUIT_KEYS:
            break
        if count >= max_frames:
            break
    if not count:
        raise ValueError("Recording ended before the first observation was saved.")
    print(f"Saved {count} RGB frames in {directory}")


def replay(directory, no_display):
    with np.load(directory / "reference.npz", allow_pickle=False) as ref:
        detector = DeskChangeDetector(ref["color_bgr"])
    counts = Counter()
    previous_state = None
    for path in sorted(directory.glob("frame_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            color = data["color_bgr"]
            state, boxes = detector.observe(color)
            counts[state] += 1
            if state != previous_state:
                print(f"frame {int(data['frame_id'])}: {state} ({float(data['color_timestamp_ms']):.0f} ms)")
                previous_state = state
            if not no_display and preview(color, state, boxes, delay=200) in QUIT_KEYS:
                break
    if not counts:
        raise ValueError("Session contains no complete recorded frames.")
    print(f"Replayed {sum(counts.values())} RGB frames: {dict(counts)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("record", help="Capture one empty reference and a test sequence")
    playback = modes.add_parser("replay", help="Repeat the experiment on a saved session")
    for mode in (capture, playback):
        mode.add_argument("directory", type=Path)
        mode.add_argument("--no-display", action="store_true")
    capture.add_argument("--serial", required=True)
    capture.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    capture.add_argument("--max-frames", type=int, default=150, help="Recorded frames after reference")
    args = parser.parse_args(argv)
    try:
        if args.mode == "replay":
            replay(args.directory, args.no_display)
        else:
            if args.directory.exists():
                raise ValueError("Choose a new session directory; this path already exists.")
            if args.max_frames < 1 or (args.no_display and args.roi is None):
                raise ValueError("Set --max-frames >= 1 and supply --roi when using --no-display.")
            with RealSenseSource(serial=args.serial, include_depth=False) as source:
                frames = source.frames()
                for _ in islice(frames, 30):
                    pass
                record_frames(frames, args.directory, args.roi, args.no_display, args.max_frames)
    except (OSError, InputSourceError, ValueError, KeyError, BadZipFile, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

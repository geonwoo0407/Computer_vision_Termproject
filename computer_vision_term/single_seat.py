"""Single fixed-camera RGB seat observations and timed confirmation."""

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice
import math
import os
from pathlib import Path

import cv2
import numpy as np

from desk import DeskChangeDetector, QUIT_KEYS, REFERENCE_SAMPLES, annotated_frame, sampled_frames


LABELS = {"empty": "미사용중", "occupied": "사용중", "away": "자리비움"}
COLORS = {"empty": (240, 178, 75), "occupied": (108, 212, 103), "away": (41, 179, 251)}
FURNITURE = {"chair", "dining table", "bench", "couch"}
WINDOW = "SeatSense single seat setup"


# Rectangles in this module use xyxy with exclusive right/bottom edges.
def intersection(a, b):
    x, y, r, d = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return (x, y, r, d) if x < r and y < d else None


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1]) if box else 0


def crop(image, box):
    x, y, r, d = box
    return image[y:d, x:r]


def anchored_in(box, roi, y_fraction=0.5):
    x = (box[0] + box[2]) / 2
    y = box[1] + (box[3] - box[1]) * y_fraction
    return roi[0] <= x < roi[2] and roi[1] <= y < roi[3]


def selected_chair(detections, roi):
    chairs = [d["bbox"] for d in detections if d["class_name"] == "chair"
              and anchored_in(d["bbox"], roi)]
    return max(chairs, key=lambda box: area(intersection(box, roi)), default=None)


@dataclass
class Evidence:
    state: str
    source: str
    box: tuple | None = None
    confidence: float | None = None

    def matches(self, other):
        if self.state != other.state:
            return False
        if self.box is None or other.box is None:
            return self.box is other.box
        # A YOLO miss can be bridged by an RGB candidate at the same spot.
        if self.source != other.source and "RGB" not in (self.source, other.source):
            return False
        return area(intersection(self.box, other.box)) >= 0.35 * max(1, min(area(self.box), area(other.box)))


@dataclass
class Track:
    evidence: Evidence
    start: float
    hits: deque = field(default_factory=deque)


class SeatConfirmation:
    """One vote per processed frame and location. Time comes from capture, not inference."""
    def __init__(self, seconds=5.0, fraction=0.8, min_hits=8):
        if not math.isfinite(seconds) or seconds < 5:
            raise ValueError("Seat confirmation requires at least 5 seconds.")
        if not 0 < fraction <= 1 or min_hits < 1:
            raise ValueError("Invalid seat vote thresholds.")
        self.seconds, self.fraction, self.min_hits = seconds, fraction, min_hits
        self.frames = deque()
        self.tracks = []
        self.state = "empty"  # The operator has just supplied an empty reference.
        self.state_since = None

    def update(self, stamp, observations):
        if not math.isfinite(stamp):
            raise ValueError("Seat observations require a finite capture timestamp.")
        if self.frames and not 0 < stamp - self.frames[-1] <= 1.0:
            if stamp < self.frames[-1]:
                self.state_since = None
            self.frames.clear()
            self.tracks.clear()
        self.frames.append(stamp)
        cutoff = stamp - self.seconds
        while self.frames[0] < cutoff:
            self.frames.popleft()
        self.tracks = [t for t in self.tracks if stamp - t.hits[-1] <= 1.0]
        current = []
        for obs in observations:
            track = next((t for t in self.tracks if t.evidence.matches(obs)), None)
            if track is None:
                track = Track(obs, stamp)
                self.tracks.append(track)
            if track.hits and track.hits[-1] == stamp:
                continue
            track.evidence = obs
            track.hits.append(stamp)
            current.append(track)
        for track in self.tracks:
            while track.hits[0] < cutoff:
                track.hits.popleft()
        best = max(current, key=lambda t: len(t.hits), default=None)
        hits = len(best.hits) if best else 0
        age = stamp - best.start if best else 0.0
        confirmed = (best is not None and age >= self.seconds and hits >= self.min_hits
                     and hits >= self.fraction * len(self.frames))
        if confirmed:
            if self.state_since is None or self.state != best.evidence.state:
                self.state_since = stamp
            self.state = best.evidence.state
        return {
            "state": self.state, "label": LABELS[self.state],
            "confirmed_this_frame": confirmed,
            "state_elapsed_seconds": round(stamp - self.state_since, 2) if self.state_since is not None else None,
            "candidate": best.evidence.state if best else None,
            "source": best.evidence.source if best else None,
            "yolo_confidence": best.evidence.confidence if best else None,
            "hits": hits, "frames": len(self.frames), "observed_seconds": round(age, 2),
            "window_seconds": self.seconds, "min_hits": self.min_hits, "min_fraction": self.fraction,
        }


class SingleSeat:
    def __init__(self, reference, desk_roi, chair_roi, initial_detections, seconds=5,
                 *, owned_pixels=None):
        self.desk, self.chair = desk_roi, chair_roi
        self.owned_pixels = owned_pixels
        self.desk_model = DeskChangeDetector(crop(reference, self.desk))
        self.ref_chair = selected_chair(initial_detections, self.chair)
        self.chair_patch = self.chair
        if self.ref_chair:
            # Compare the interior in chair-relative coordinates as the chair moves.
            self.chair_patch = intersection(self.chair, self.ref_chair)
        self.chair_model = DeskChangeDetector(crop(reference, self.chair_patch))
        self.confirmation = SeatConfirmation(seconds)

    def chair_view(self, box):
        if self.ref_chair is None:
            return self.chair
        if box is None:
            return None
        x, y, r, d = self.ref_chair
        scale_x = (box[2] - box[0]) / (r - x)
        scale_y = (box[3] - box[1]) / (d - y)
        a, b, c, e = self.chair_patch
        mapped = (round(box[0] + (a-x)*scale_x), round(box[1] + (b-y)*scale_y),
                  round(box[0] + (c-x)*scale_x), round(box[1] + (e-y)*scale_y))
        if area(mapped) and area(intersection(mapped, self.chair)) >= .75 * area(mapped):
            return mapped
        return None

    def rgb_evidence(self, model, frame, roi, occluders):
        if roi is None:
            model.streak.fill(0)
            return [], False
        patch = crop(frame, roi)
        height, width = model.color.shape[:2]
        mask = np.zeros(patch.shape[:2], np.uint8)
        if self.owned_pixels is not None:
            mask[~crop(self.owned_pixels, roi)] = 1
        for box in occluders:
            part = intersection(roi, box)
            if part:
                x, y, r, d = part
                mask[y-roi[1]:d-roi[1], x-roi[0]:r-roi[0]] = 1
        scale_x, scale_y = patch.shape[1] / width, patch.shape[0] / height
        if patch.shape[:2] != (height, width):
            patch = cv2.resize(patch, (width, height))
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        ignore = mask.astype(bool)
        status, boxes = model.observe(patch, local_contrast=True, ignore=ignore)
        evidence = []
        for x, y, w, h in boxes:
            evidence.append(Evidence("away", "RGB", (
                roi[0] + round(x*scale_x), roi[1] + round(y*scale_y),
                roi[0] + round((x+w)*scale_x), roi[1] + round((y+h)*scale_y))))
        return evidence, status == "no_change" and ignore.mean() < 0.5

    def observe(self, frame, detections, stamp, *, other_blockers=()):
        people, items, blockers, chairs = [], [], list(other_blockers), []
        for detection in detections:
            name, box = detection["class_name"], tuple(detection["bbox"])
            if name == "person":
                blockers.append(box)
                if anchored_in(box, self.chair, 0.60):
                    people.append(Evidence("occupied", name, box, detection["confidence"]))
            elif name == "chair":
                chairs.append(box)
            elif name not in FURNITURE and (anchored_in(box, self.desk) or anchored_in(box, self.chair)):
                items.append(Evidence("away", name, box, detection["confidence"]))
        current_chair = selected_chair(detections, self.chair)
        desk_blockers = blockers + chairs
        if self.ref_chair:
            desk_blockers.append(self.ref_chair)
        desk_items, clear_desk = self.rgb_evidence(self.desk_model, frame, self.desk, desk_blockers)
        chair_items, clear_chair = self.rgb_evidence(
            self.chair_model, frame, self.chair_view(current_chair),
            blockers + (chairs if self.ref_chair is None else []))
        rgb_items = desk_items + chair_items
        candidates = people or (items + rgb_items)
        if not candidates and clear_desk and clear_chair:
            candidates = [Evidence("empty", "clear")]
        result = self.confirmation.update(stamp, candidates)
        result.update(desk_roi=self.desk, chair_roi=self.chair, rgb_boxes=[e.box for e in rgb_items])
        return result


def empty_reference(packets, view, rois_xywh, window, no_display):
    print("Keep the selected desks and chairs empty for ~3 s.", flush=True)
    samples = []
    for packet in islice(sampled_frames(packets, view.frame.shape), REFERENCE_SAMPLES):
        samples.append(packet.frame.copy())
        if not no_display:
            msg = f"Empty reference {len(samples)}/{REFERENCE_SAMPLES} | Q: quit"
            cv2.imshow(window, annotated_frame(packet.frame, msg, rois_xywh))
            if cv2.waitKey(1) & 255 in QUIT_KEYS:
                return None
    if len(samples) < REFERENCE_SAMPLES:
        raise ValueError("Input ended before the empty-seat reference was ready.")
    return np.median(samples, axis=0).astype(np.uint8)


def prepare_single_seat(packets, detector, desk_roi, chair_roi, no_display, seconds):
    for view in packets:
        if desk_roi is None:
            cv2.imshow(WINDOW, annotated_frame(view.frame, "Empty desk and chair | F: freeze   Q: quit", []))
            key = cv2.waitKey(1) & 255
            if key in QUIT_KEYS:
                return None
            if key not in (ord("f"), ord("F")):
                continue
            choices = []
            for name in ("DESK surface", "CHAIR and seated torso"):
                guide = annotated_frame(view.frame, f"{name} | Drag, Enter to accept, C to retry", choices)
                rect = cv2.selectROI(WINDOW, guide, showCrosshair=False, fromCenter=False)
                if min(rect[2:]) < 4:
                    break
                choices.append(rect)
            if len(choices) < 2:
                continue
            desk_roi, chair_roi = choices
        boxes = []
        height, width = view.frame.shape[:2]
        for x, y, w, h in (desk_roi, chair_roi):
            if min(w, h) < 4 or min(x, y) < 0 or x+w > width or y+h > height:
                raise ValueError(f"Seat ROI must fit inside {width}x{height}, minimum size 4x4.")
            boxes.append((x, y, x+w, y+h))
        reference = empty_reference(packets, view, [desk_roi, chair_roi], WINDOW, no_display)
        if reference is None:
            return None
        initial = detector.process_frame(reference, view.camera_id, view.timestamp)["detections"]
        if not no_display:
            cv2.destroyWindow(WINDOW)
        return SingleSeat(reference, *boxes, initial, seconds)
    raise ValueError("Input ended before the empty-seat reference was ready.")


@lru_cache(maxsize=3)
def status_font(size=24):
    from PIL import ImageFont
    for path in (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
                 Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
                 Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return None


def draw_regions(frame, seat, prefix=""):
    for box, color, name in [(seat["desk_roi"], (215, 214, 75), "desk"),
                              (seat["chair_roi"], (238, 140, 214), "seat")]:
        x, y, r, d = box
        cv2.rectangle(frame, (x, y), (r-1, d-1), color, 1)
        cv2.putText(frame, prefix + name, (x+2, y+14), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for x, y, r, d in seat["rgb_boxes"]:
        cv2.rectangle(frame, (x, y), (r-1, d-1), (0, 165, 255), 2)
        cv2.putText(frame, "RGB", (x, max(14, y-3)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 165, 255), 1)


def draw_status(frame, seat):
    from PIL import Image, ImageDraw
    draw_regions(frame, seat)
    panel = Image.new("RGB", (max(frame.shape[1], 430), 88), (24, 31, 43))
    ink = COLORS[seat["state"]][::-1]
    pen = ImageDraw.Draw(panel)
    font = status_font()
    title = seat["label"] if font else seat["state"].upper()
    elapsed = seat["state_elapsed_seconds"]
    duration = f"{elapsed:.1f}s" if elapsed is not None else "--"
    pen.text((10, 0), f"{title}   {duration}", fill=ink, font=font)
    score = seat["yolo_confidence"]
    detail = seat["source"] or "uncertain"
    if score is not None:
        detail += f" {score:.2f}"
    cv_panel = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    lines = [f"Candidate: {seat['candidate'] or 'hold'}  |  Evidence: {detail}",
             f"Hits {seat['hits']}/{seat['frames']} in {seat['window_seconds']:g}s  |  Candidate {seat['observed_seconds']:.1f}s"]
    for i, line in enumerate(lines):
        cv2.putText(cv_panel, line, (10, 53 + 23*i), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (225, 230, 236), 1, cv2.LINE_AA)
    return cv2.vconcat([cv2.copyMakeBorder(frame, 0, 0, 0, cv_panel.shape[1]-frame.shape[1],
                                           cv2.BORDER_CONSTANT), cv_panel])

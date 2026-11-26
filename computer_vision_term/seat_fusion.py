"""Two-view RGB observations for seats with a shared tabletop and separate chair regions."""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from desk import DeskChangeDetector


PATCH_SIZE = (160, 96)
PATCH_CORNERS = np.float32([[0, 0], [159, 0], [159, 95], [0, 95]])
MAX_PAIR_SKEW = 0.5  # host monotonic seconds, measured at capture
MAX_SAMPLE_GAP = 1.0
PRESENCE_SECONDS = 2.0
EMPTY_SECONDS = 5.0


class SeatView:
    def __init__(self, reference, corners, chair_region):
        self.shape = reference.shape
        self.corners = np.asarray(corners, np.float32)
        height, width = self.shape[:2]
        if (self.corners.shape != (4, 2) or not np.isfinite(self.corners).all()
                or not cv2.isContourConvex(self.corners) or cv2.contourArea(self.corners) < 32
                or (self.corners < 0).any() or (self.corners > [width-1, height-1]).any()):
            raise ValueError("Select four distinct tabletop corners in perimeter order.")
        x, y, w, h = chair_region
        if min(x, y) < 0 or min(w, h) < 4 or x+w > width or y+h > height:
            raise ValueError("Chair region must lie inside the reference image.")
        self.chair_region = (x, y, w, h)
        self.to_table = cv2.getPerspectiveTransform(self.corners, PATCH_CORNERS)
        self.detector = DeskChangeDetector(self.rectify(reference))

    def rectify(self, frame):
        if frame.shape != self.shape:
            raise ValueError("Camera resolution differs from the calibration.")
        return cv2.warpPerspective(frame, self.to_table, PATCH_SIZE)

    def contains(self, point, allow_table):
        x, y, w, h = self.chair_region
        px, py = point
        in_chair = x <= px < x+w and y <= py < y+h
        return in_chair or (allow_table and cv2.pointPolygonTest(self.corners, (px, py), False) >= 0)


@dataclass
class Observation:
    person: bool | None
    known_item: bool | None
    table: str
    changes: list  # tabletop patch coordinates (x, y, width, height)


def observe_view(frame, seats, detections):
    evidence = {seat: {"person": False, "known_item": False} for seat in seats}
    occluders = np.zeros(frame.shape[:2], np.uint8)
    for detection in detections:
        name = detection["class_name"]
        x1, y1, x2, y2 = detection["bbox"]
        if name in ("chair", "person"):
            cv2.rectangle(occluders, (x1, y1), (x2, y2), 255, -1)
        if name == "chair":
            continue
        center = ((x1+x2)/2, (y1+y2)/2)
        matches = [seat for seat, view in seats.items() if view.contains(center, name != "person")]
        field = "person" if name == "person" else "known_item"
        for seat in matches:
            if evidence[seat][field] is not True:
                evidence[seat][field] = True if len(matches) == 1 else None

    result = {}
    for seat, view in seats.items():
        state, changes = view.detector.observe(view.rectify(frame))
        blocked = cv2.warpPerspective(occluders, view.to_table, PATCH_SIZE, flags=cv2.INTER_NEAREST)
        visible = [(x, y, w, h) for x, y, w, h in changes
                   if not blocked[y:y+h, x:x+w].any()]
        # A change spanning most of the patch is also consistent with lighting or a moved camera.
        if any(w*h > 0.6 * blocked.size for x, y, w, h in changes):
            state, visible = "uncertain", []
        elif (changes and not visible) or (state == "no_change" and np.count_nonzero(blocked) > blocked.size * 0.15):
            state = "occluded"
        result[seat] = Observation(**evidence[seat], table=state, changes=visible)
    return result


def same_table_region(a, b):
    x, y, w, h = a
    u, v, s, t = b
    width = max(0, min(x+w, u+s) - max(x, u))
    height = max(0, min(y+h, v+t) - max(y, v))
    return width * height >= 0.25 * min(w*h, s*t)


def combine(left, right):
    people = (left.person, right.person)
    occupied = any(people)
    if not occupied and None in people:
        return None
    if left.known_item is True or right.known_item is True:
        belongings = True
    elif any(same_table_region(a, b) for a in left.changes for b in right.changes):
        belongings = True
    elif (left.table == right.table == "no_change"
          and left.known_item is False and right.known_item is False):
        belongings = False
    else:
        return None
    return occupied, belongings


class SeatFusion:
    """Consume each fresh pair once; missing or uncertain evidence breaks a streak."""

    def __init__(self, serials):
        if len(serials) != 2 or len(set(serials)) != 2:
            raise ValueError("Two distinct camera serials are required.")
        self.serials = tuple(serials)
        self.frames = {}
        self.last_seen = {}
        self.streaks = {}

    def update(self, serial, captured, views, now):
        if serial not in self.serials:
            raise ValueError("Unexpected camera for this calibration.")
        if not math.isfinite(captured) or not 0 <= now-captured <= MAX_SAMPLE_GAP:
            return {}
        if captured <= self.last_seen.get(serial, -math.inf):
            return {}
        self.last_seen[serial] = captured
        self.frames[serial] = (captured, views)
        if len(self.frames) < 2:
            return {}
        first, second = (self.frames[key] for key in self.serials)
        if abs(first[0]-second[0]) > MAX_PAIR_SKEW or now-min(first[0], second[0]) > MAX_SAMPLE_GAP:
            older = min(self.frames, key=lambda key: self.frames[key][0])
            del self.frames[older]
            return {}
        self.frames.clear()
        stamp = max(first[0], second[0])
        result = {}
        for seat in first[1].keys() & second[1].keys():
            state = combine(first[1][seat], second[1][seat])
            if state is None:
                self.streaks.pop(seat, None)
                result[seat] = ("uncertain", None)
                continue
            previous, start, last, count = self.streaks.get(seat, (None, stamp, stamp, 0))
            if previous != state or stamp-last > MAX_SAMPLE_GAP:
                start, count = stamp, 0
            self.streaks[seat] = (state, start, stamp, count+1)
            delay = PRESENCE_SECONDS if any(state) else EMPTY_SECONDS
            stable = count+1 >= 3 and stamp-start >= delay
            result[seat] = ("stable" if stable else "pending", state)
        return result

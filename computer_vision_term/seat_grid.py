"""Six independent seat observations from one fixed RGB camera."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from desk import QUIT_KEYS
from single_seat import COLORS, FURNITURE, SingleSeat, draw_regions, empty_reference, status_font


ROWS, COLS = 2, 3
SLOTS = ROWS * COLS
WINDOW = "SeatSense 3 x 2 setup"
DEFAULT_LAYOUT = Path("output/seat_grid_3x2.json")


def slot_name(index):
    column = ("LEFT", "CENTER", "RIGHT")[index % COLS]
    return f"{index+1} ({'FAR' if index < COLS else 'NEAR'} {column})"


def ownership(shape, regions):
    """In overlaps, give each pixel to the nearest ROI center in ROI-relative units."""
    owners = np.full(shape, -1, dtype=np.int8)
    distances = np.full(shape, np.inf, dtype=np.float32)
    for idx, boxes in enumerate(regions):
        for x, y, r, d in boxes:
            yy, xx = np.ogrid[y:d, x:r]
            distance = ((2*xx+1-x-r)/(r-x))**2 + ((2*yy+1-y-d)/(d-y))**2
            local = distances[y:d, x:r]
            nearer = distance < local
            owners[y:d, x:r][nearer] = idx
            local[nearer] = distance[nearer]
    return owners


class SeatGrid:
    def __init__(self, reference, regions, detections, seconds):
        shape = reference.shape[:2]
        self.owners = ownership(shape, regions)
        self.chair_owners = ownership(shape, [[chair] for desk, chair in regions])
        assigned, _ = self.assign(detections)
        self.seats = [SingleSeat(reference, desk, chair, assigned[i], seconds,
                                 owned_pixels=(self.owners == i))
                      for i, (desk, chair) in enumerate(regions)]

    def assign(self, detections):
        groups = [[] for _ in range(SLOTS)]
        blockers = [[] for _ in range(SLOTS)]
        height, width = self.owners.shape
        for obj in detections:
            name, box = obj["class_name"], obj["bbox"]
            if name in FURNITURE and name != "chair":
                continue
            x, y, r, d = box
            px = (x+r)//2
            py = int(y + (d-y) * (.60 if name == "person" else .50))
            owners = self.chair_owners if name in ("person", "chair") else self.owners
            owner = int(owners[py, px]) if 0 <= px < width and 0 <= py < height else -1
            if owner >= 0:
                groups[owner].append(obj)
            # The same box can obscure other seats without counting as their person/bag.
            for i in range(SLOTS):
                if i != owner:
                    blockers[i].append(box)
        return groups, blockers

    def observe(self, frame, detections, stamp):
        groups, blockers = self.assign(detections)
        results = []
        for i, seat in enumerate(self.seats):
            result = seat.observe(frame, groups[i], stamp, other_blockers=blockers[i])
            result.update(slot=i+1, row=i//COLS+1, column=i%COLS+1)
            results.append(result)
        return results


def read_layout(path, view):
    data = json.loads(path.read_text(encoding="utf-8"))
    size = [view.frame.shape[1], view.frame.shape[0]]
    if (not isinstance(data, dict) or data.get("version") != 1 or data.get("image_size") != size or
            (view.serial_number is not None and data.get("serial") != view.serial_number)):
        raise ValueError(f"{path}: camera, image size or layout version differs; choose a new --grid-layout path.")
    seats = data.get("seats")
    if not isinstance(seats, list) or len(seats) != SLOTS:
        raise ValueError(f"{path}: expected six seats in far-to-near, left-to-right order.")
    regions = []
    for i, seat in enumerate(seats):
        if not isinstance(seat, dict) or seat.get("slot") != i+1:
            raise ValueError(f"{path}: expected slot {i+1}.")
        boxes = []
        for key in ("desk", "chair"):
            box = seat.get(key)
            if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
                raise ValueError(f"{path}: invalid {key} rectangle in slot {i+1}.")
            x, y, r, d = box
            if min(x, y) < 0 or r > size[0] or d > size[1] or min(r-x, d-y) < 4:
                raise ValueError(f"{path}: {key} rectangle in slot {i+1} must fit inside the image.")
            boxes.append(tuple(box))
        regions.append(boxes)
    return regions


def save_layout(path, view, regions):
    data = {"version": 1, "image_size": [view.frame.shape[1], view.frame.shape[0]],
            "serial": view.serial_number,
            "seats": [dict(slot=i+1, desk=desk, chair=chair) for i, (desk, chair) in enumerate(regions)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
            json.dump(data, tmp, indent=2)
            tmp.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def setup_guide(frame, boxes):
    guide = frame.copy()
    for j, (x, y, r, d) in enumerate(boxes):
        desk = j % 2 == 0
        color = (215, 214, 75) if desk else (238, 140, 214)
        cv2.rectangle(guide, (x, y), (r-1, d-1), color, 1)
        cv2.putText(guide, f"{j//2+1}{'D' if desk else 'C'}", (x+1, min(y+14, d-1)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return guide


def instruction(frame, message):
    # Keep the header out of the image so even the far desktops can be selected.
    canvas = cv2.copyMakeBorder(frame, 0, 32, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(canvas, message, (6, frame.shape[0]+21), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def select_regions(frame):
    boxes = []
    while len(boxes) < SLOTS * 2:
        idx = len(boxes)
        part = "DESK surface" if idx % 2 == 0 else "CHAIR + torso"
        guide = instruction(setup_guide(frame, boxes), f"{slot_name(idx//2)}: {part} | drag, Enter | C: back")
        x, y, w, h = cv2.selectROI(WINDOW, guide, showCrosshair=False, fromCenter=False)
        if w == 0 or h == 0:
            if not boxes:
                return None
            boxes.pop()
            continue
        bottom = min(y+h, frame.shape[0])
        if w >= 4 and bottom-y >= 4:
            boxes.append((x, y, x+w, bottom))
    return list(zip(boxes[::2], boxes[1::2]))


def prepare_grid(packets, detector, layout_path, no_display, seconds):
    path = Path(layout_path)
    first = next(packets, None)
    if first is None:
        raise ValueError("No frames available for seat setup.")
    saved = read_layout(path, first) if path.exists() else None
    if no_display and saved is None:
        raise ValueError(f"Register the six seats with the preview first: {path}")
    view, regions = first, saved
    if not no_display:
        saved_boxes = [box for pair in saved or [] for box in pair]
        while view is not None:
            prompt = "F: use saved regions | R: redraw | Q: quit" if saved else "Empty all 6 seats | F: freeze | Q: quit"
            cv2.imshow(WINDOW, instruction(setup_guide(view.frame, saved_boxes), prompt))
            key = cv2.waitKey(1) & 255
            if key in QUIT_KEYS:
                return None
            if key in (ord("f"), ord("F"), ord("r"), ord("R")):
                if saved and key in (ord("f"), ord("F")):
                    break
                regions = select_regions(view.frame)
                if regions:
                    break
            view = next(packets, None)
        if view is None:
            raise ValueError("Input ended before seat setup was complete.")
    xywh = [(x, y, r-x, d-y) for pair in regions for x, y, r, d in pair]
    reference = empty_reference(packets, view, xywh, WINDOW, no_display)
    if reference is None:
        return None
    detections = detector.process_frame(reference, view.camera_id, view.timestamp)["detections"]
    if regions is not saved:
        save_layout(path, view, regions)
        print(f"Saved six-seat regions to {path}", flush=True)
    if not no_display:
        cv2.destroyWindow(WINDOW)
    return SeatGrid(reference, regions, detections, seconds)


def draw_grid(frame, seats):
    width = max(540, frame.shape[1])
    panel = Image.new("RGB", (width, 172), (24, 31, 43))
    pen = ImageDraw.Draw(panel)
    font = status_font()
    small = status_font(13)
    for seat in seats:
        draw_regions(frame, seat, f"{seat['slot']} ")
        x = (seat["column"] - 1) * width // COLS + 8
        y = (seat["row"] - 1) * 86
        state = seat["label"] if font else seat["state"].upper()
        pen.text((x, y), f"{seat['slot']}  {state}", fill=COLORS[seat["state"]][::-1], font=font)
        source = seat["source"] or ("관측 보류" if small else "hold")
        if seat["yolo_confidence"] is not None:
            source += f" {seat['yolo_confidence']:.2f}"
        elapsed = seat["state_elapsed_seconds"]
        duration = f"{elapsed:.1f}s" if elapsed is not None else "--"
        elapsed_label = "상태 경과" if small else "State"
        app_number = f"   App #{seat['app_seat_id']}" if "app_seat_id" in seat else ""
        pen.text((x, y+31), f"{elapsed_label}  {duration}{app_number}", fill="white", font=small)
        pen.text((x, y+49), source, fill=(206, 214, 229), font=small)
        candidate = "후보" if small else "Candidate"
        pen.text((x, y+65), f"{candidate} {seat['observed_seconds']:.1f}s   Hits {seat['hits']}/{seat['frames']}",
                 fill=(206, 214, 229), font=small)
    padded = cv2.copyMakeBorder(frame, 0, 0, 0, width-frame.shape[1], cv2.BORDER_CONSTANT)
    return cv2.vconcat([padded, cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)])

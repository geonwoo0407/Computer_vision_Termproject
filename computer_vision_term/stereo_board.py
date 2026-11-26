"""Interactive RealSense checkerboard capture and independent RGB-D validation."""

import argparse
from contextlib import contextmanager
import csv
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from camera import InputSourceError, MultiRealSenseSource, list_realsense_devices
from main import load_config
from stereo_geometry import board_points, calculate_3d_distance, camera_matrix, fit_stereo, ideal_pixels, load_calibration


WINDOW = "SeatSense stereo calibration"
MAX_SKEW_S = 0.080


def detect_board(frame, pattern):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
    ok, corners = cv2.findChessboardCornersSB(gray, tuple(pattern), flags=flags)
    # OpenCV 4/5 bindings return (N,1,2) or (N,2) depending on version.
    return corners.reshape(-1, 1, 2) if ok else None


def print_pattern(path, pattern, square_mm):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    board_points(pattern, square_mm)
    width, height = (n + 1 for n in pattern)
    paper = (297, 210) if width >= height else (210, 297)
    x, y = (paper[0]-width*square_mm)/2, (paper[1]-height*square_mm)/2
    if min(x, y) < 10:
        raise ValueError("This board needs larger paper; use a smaller square size for the A4 PDF.")
    fig = Figure(figsize=(paper[0]/25.4, paper[1]/25.4))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, paper[0]), ylim=(0, paper[1]), aspect="equal")
    ax.axis("off")
    for row in range(height):
        for col in range(width):
            if (row+col) % 2 == 0:
                ax.add_patch(Rectangle((x+col*square_mm, y+row*square_mm), square_mm,
                                       square_mm, facecolor="black", edgecolor="none"))
    for tag, offset in (("A", 1), ("B", width-1)):
        ax.text(x+offset*square_mm, y+height*square_mm+4, tag, ha="center")
    ax.text(paper[0]/2, y-6, f"{pattern[0]} x {pattern[1]} inner corners | {square_mm:g} mm | Print at 100%",
            ha="center", va="center", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        fig.savefig(stream, format="pdf")
    print(f"A4 board saved to {path}")


def label(image, text, point=(8, 24), scale=0.55):
    for color, thickness in (((0, 0, 0), 3), ((60, 245, 245), 1)):
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def review_pair(packets, pattern):
    frames = [p.frame for p in packets]
    corners = [detect_board(frame, pattern) for frame in frames]
    cols, rows = pattern
    while True:
        images = []
        for frame, points in zip(frames, corners):
            img = frame.copy()
            if points is not None:
                cv2.drawChessboardCorners(img, tuple(pattern), points, True)
                for n, xy in enumerate(points.reshape(-1, 2)):
                    anchor = tuple(np.rint(xy).astype(int))
                    tag = {0: "A:0", cols-1: f"B:{cols-1}"}.get(n, str(n))
                    label(img, tag, anchor, 0.36)
            label(img, "Board OK" if points is not None else "Board missing")
            images.append(img)
        preview = np.hstack(images)
        if all(points is not None for points in corners):
            hint = "Same physical A and B? Enter: accept   1/2: reorder camera   C: discard   Q: quit"
        else:
            hint = f"Need {cols}x{rows} inner corners in both views. C: live/reframe   Q: quit/change --inner"
        label(preview, hint, (8, preview.shape[0]-12))
        cv2.imshow(WINDOW, preview)
        key = cv2.waitKey(0) & 255
        if key in (27, ord("q")):
            raise KeyboardInterrupt
        if key == ord("c"):
            return None
        if key in (10, 13) and all(p is not None for p in corners):
            return tuple(corners)
        if key in (ord("1"), ord("2")):
            index = key - ord("1")
            if corners[index] is not None:
                grid = corners[index].reshape(rows, cols, 2)
                grid = np.rot90(grid, 1 if cols == rows else 2)
                corners[index] = np.ascontiguousarray(grid.reshape(-1, 1, 2))


def duplicate_pose(samples, points):
    # 두 카메라에서 모두 제자리인 경우, 전체 인덱스 방향이 뒤집혔어도 같은 자세다.
    def distance(a, b):
        return float(np.sqrt(np.mean((a-b)**2)))
    return any(max(min(distance(a, b), distance(a[::-1], b)) for a, b in zip(pair, points)) < 4
               for pair in samples)


def validate_calibration(calibration, packets, corners, pose_id):
    """Compare independent measurements of each physical corner in camera 1 space."""
    rows = []
    serials = calibration.serials
    board = board_points(calibration.metadata["inner_corners"], calibration.metadata["square_mm"])
    intr = calibration.get_camera_intrinsics(serials[0])
    ok, rotation, translation = cv2.solvePnP(board, ideal_pixels(intr, corners[0]), camera_matrix(intr), None)
    normal = cv2.Rodrigues(rotation)[0][:, 2] if ok else None
    angle = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))) if ok else None
    reprojection = np.full(len(board), np.nan)
    if ok:
        # 검증 자세는 카메라 1의 코너만으로 구한 뒤 카메라 2에서 독립적으로 비교한다.
        xyz1 = board @ cv2.Rodrigues(rotation)[0].T + translation.ravel()
        transform = calibration.cam1_from_cam2
        xyz2 = (xyz1-transform[:3,3]) @ transform[:3,:3]
        intr2 = calibration.get_camera_intrinsics(serials[1])
        uv2 = cv2.projectPoints(xyz2, np.zeros(3), np.zeros(3), camera_matrix(intr2), None)[0]
        measured = ideal_pixels(intr2, corners[1])
        reprojection = np.linalg.norm(uv2.reshape(-1, 2)-measured.reshape(-1, 2), axis=1)
    for i, points in enumerate(zip(*corners)):
        positions = []
        row = dict(pose=pose_id, corner=i, tilt_cam1_deg=angle, holdout_reprojection_px=reprojection[i])
        for j, (p, corner) in enumerate(zip(packets, points), 1):
            u, v = corner.ravel()
            x, y = np.rint([u, v]).astype(int)
            depth = p.depth_m
            local = depth[max(y-2,0):y+3, max(x-2,0):x+3] if depth is not None else np.array([])
            valid = local[np.isfinite(local) & (local > 0)]
            z = float(np.median(valid)) if valid.size >= 5 else None
            observation = calibration.pixel_to_3d(p.serial_number, (u,v), z, p.received_monotonic)
            row.update({f"host_s_{j}": p.received_monotonic, f"depth_m_{j}": z,
                        f"depth_pixels_{j}": valid.size})
            positions.append(observation["position_common_m"])
            for axis, value in zip("xyz", observation["position_common_m"] or [None]*3):
                row[f"cam{j}_common_{axis}_m"] = value
        error = None if None in positions else np.subtract(positions[1], positions[0])
        row["valid"] = error is not None
        for axis, value in zip("xyz", error if error is not None else [None]*3):
            row[f"d{axis}_m"] = value
        row["error_m"] = calculate_3d_distance(*positions) if error is not None else None
        rows.append(row)
    return rows


def validation_summary(rows):
    valid = [r for r in rows if r["valid"]]
    if not valid:
        return "No paired valid depth samples"
    e = np.asarray([r["error_m"] for r in valid])
    return (f"{len(valid)}/{len(rows)} points | mean {1000*e.mean():.1f} mm | "
            f"max {1000*e.max():.1f} mm | RMSE {1000*np.sqrt(np.mean(e**2)):.1f} mm")


def write_report(path, rows):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    valid = [r for r in rows if r["valid"]]
    if valid:
        fig = Figure(figsize=(9, 4), layout="constrained")
        FigureCanvasAgg(fig)
        for ax, key, title in zip(fig.subplots(1, 2), ["depth_m_1", "tilt_cam1_deg"],
                                  ["Camera 1 depth (m)", "Board tilt (deg)"]):
            ax.scatter([r[key] for r in valid], [1000*r["error_m"] for r in valid], s=7, alpha=.5)
            ax.set(xlabel=title, ylabel="3D discrepancy (mm)")
        fig.savefig(path.with_suffix(".png"), dpi=150)
    print(validation_summary(rows), flush=True)


@contextmanager
def validation_report(path, records):
    try:
        yield
    finally:
        if path is not None and records:
            write_report(path, records)


def run(args):
    config = load_config(args.config)
    settings = config.get("checkerboard", {})
    checking = args.mode == "validate"
    calibration = load_calibration(args.file) if checking else None
    if checking:
        pattern = tuple(calibration.metadata["inner_corners"])
        serials = calibration.serials
        if args.csv.suffix.lower() != ".csv" or args.csv.exists() or args.csv.with_suffix(".png").exists():
            raise ValueError("Choose a new .csv report path (including its .png companion).")
        print(f"Loaded {args.file}, fit RMS {calibration.metadata['reprojection_rms_px']:.3f} px")
    else:
        if args.file.exists():
            raise ValueError("Choose a new calibration output path.")
        pattern = tuple(args.inner or settings.get("inner_corners", [9, 6]))
        square = args.square_mm if args.square_mm is not None else settings.get("square_mm")
        if square is None:
            raise ValueError("Measure the printed square and supply --square-mm.")
        board_points(pattern, square)
        target = settings.get("target_pairs", 20) if args.pairs is None else args.pairs
        if target < 6:
            raise ValueError("Use at least 6 pairs; 20 varied poses are recommended.")
        serials = args.serials
    available = [d.serial_number for d in list_realsense_devices()]
    serials = tuple(serials if serials else sorted(available))
    if len(serials) != 2 or len(set(serials)) != 2 or not set(serials).issubset(available):
        raise ValueError(f"Need two distinct connected cameras. Detected serials: {available}")
    print(f"Camera 1 (common): {serials[0]} | Camera 2: {serials[1]}", flush=True)
    width, height, fps = (config["realsense_"+k] for k in ("width","height","fps"))
    samples, records, latest, previous, rates, intrinsics = [], [], {}, {}, {}, {}
    status = "Hold board still, F: freeze both views"
    pose = 0
    with validation_report(args.csv if checking else None, records), MultiRealSenseSource(
            list(serials), width=width, height=height, fps=fps) as cameras:
        scales = {s.serial: s.depth_scale_m for s in cameras.sources}
        frame_rates = {s.serial: s.fps for s in cameras.sources}
        for packet in cameras.frames():
            serial, stamp = packet.serial_number, packet.received_monotonic
            if serial not in intrinsics:
                intrinsics[serial] = packet.intrinsics
                ideal_pixels(packet.intrinsics, [[0., 0.]])
                if checking:
                    calibration.check_profile(serial, packet.intrinsics, frame_rates[serial])
            if serial in previous and stamp > previous[serial]:
                rates[serial] = .85*rates.get(serial,fps) + .15/(stamp-previous[serial])
            previous[serial], latest[serial] = stamp, packet
            if len(latest) != 2:
                continue
            pair = [latest[s] for s in serials]
            skew = abs(pair[0].received_monotonic-pair[1].received_monotonic)
            rgb, depths = [], []
            for p in pair:
                view = p.frame.copy()
                label(view, f"{p.serial_number} | {rates.get(p.serial_number, 0):.1f} FPS")
                rgb.append(view)
                depth = np.nan_to_num(p.depth_m, nan=0, posinf=0, neginf=0)
                preview = cv2.applyColorMap(np.uint8(np.clip(depth*65, 0, 255)), cv2.COLORMAP_TURBO)
                preview[depth <= 0] = 0
                depths.append(cv2.resize(preview, (view.shape[1], view.shape[0]//3)))
            display = np.vstack([np.hstack(rgb), np.hstack(depths)])
            count = f"poses {pose}" if checking else f"pairs {len(samples)}/{target}"
            label(display, f"{count} | host skew {1000*skew:.0f} ms | F: freeze  X: reset  Q: finish", (8, 52))
            label(display, status, (8, display.shape[0]-12))
            if calibration:
                label(display, f"RMS {calibration.metadata['reprojection_rms_px']:.2f}px | cam2 origin in cam1 (m): "
                      + np.array2string(calibration.cam1_from_cam2[:3,3], precision=3), (8, 78))
                for line, rotation_row in enumerate(calibration.cam1_from_cam2[:3,:3]):
                    label(display, "R " + np.array2string(rotation_row, precision=3), (8, 99 + 18*line), .43)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 255
            if key in (27, ord("q")):
                break
            if key == ord("x"):
                samples.clear()
                records.clear()
                pose = 0
                if not checking:
                    calibration = None
                status = "Current session cleared"
            if not checking and key == ord("s") and calibration:
                calibration.save(args.file)
                print(f"Saved {args.file}", flush=True)
                return
            if not checking and key == ord("g"):
                try:
                    calibration = fit_stereo(serials, intrinsics, samples, pattern, square, frame_rates, scales)
                except (ValueError, cv2.error) as exc:
                    status = "Fit failed; see terminal. Capture new poses or X: reset"
                    print(exc, flush=True)
                    continue
                status = "Fit ready. S: save   F: add pose"
                print(calibration.cam1_from_cam2, calibration.metadata, flush=True)
            if key != ord("f"):
                continue
            if skew > MAX_SKEW_S or time.monotonic()-min(p.received_monotonic for p in pair) > .4:
                status = "Pair too far apart or stale. Hold still and press F again."
                continue
            points = review_pair(pair, pattern)
            latest.clear()
            if points is None:
                status = "Pair discarded. F: freeze again"
                continue
            if checking:
                pose += 1
                rows = validate_calibration(calibration, pair, points, pose)
                records.extend(rows)
                status = validation_summary(rows)
                example = next((r for r in rows if r["valid"]), None)
                if example:
                    xyz = [round(example[f"cam2_common_{axis}_m"], 3) for axis in "xyz"]
                    status += f" | corner {example['corner']} XYZ {xyz} m"
                print(status, flush=True)
            elif duplicate_pose(samples, points):
                status = "Similar pose. Move or tilt the board."
            else:
                samples.append(points)
                calibration = None
                status = "Pose added. F: next pose   G: fit   S: save fitted result"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    sub = parser.add_subparsers(dest="mode", required=True)
    pattern = sub.add_parser("pattern", help="Write a print-to-scale A4 checkerboard PDF")
    pattern.add_argument("file", type=Path)
    pattern.add_argument("--inner", type=int, nargs=2, default=[9, 6])
    pattern.add_argument("--square-mm", type=float, default=25.)
    capture = sub.add_parser("capture")
    capture.add_argument("file", type=Path)
    capture.add_argument("--serials", nargs=2)
    capture.add_argument("--inner", type=int, nargs=2, metavar=("COLS", "ROWS"))
    capture.add_argument("--square-mm", type=float)
    capture.add_argument("--pairs", type=int)
    validate = sub.add_parser("validate")
    validate.add_argument("file", type=Path)
    validate.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.mode == "pattern":
            print_pattern(args.file, args.inner, args.square_mm)
        else:
            run(args)
    except (InputSourceError, OSError, ValueError, KeyError, cv2.error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

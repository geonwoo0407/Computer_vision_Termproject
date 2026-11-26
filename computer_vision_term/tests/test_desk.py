import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from camera import FramePacket
from desk import DeskChangeDetector, main, record_frames


class DeskExperimentTests(unittest.TestCase):
    def setUp(self):
        self.bgr = np.full((48, 64, 3), 180, np.uint8)
        self.patch = np.s_[12:30, 12:30]

    def test_shadow_and_thin_colored_object(self):
        detector = DeskChangeDetector(self.bgr)
        shadow = self.bgr.copy()
        shadow[self.patch] = 85
        for _ in range(5):
            self.assertEqual(detector.observe(shadow), ("uncertain", []))
        cloth = self.bgr.copy()
        cloth[self.patch] = [90, 135, 190]
        for _ in range(2):
            self.assertEqual(detector.observe(cloth)[0], "uncertain")
        for _ in range(6):
            state, boxes = detector.observe(cloth)
            self.assertEqual(state, "object_candidate")
            self.assertEqual(len(boxes), 1)
        self.assertEqual(detector.observe(self.bgr), ("no_change", []))

    def test_legacy_rgbd_recording_ignores_depth_changes(self):
        depth = np.ones(self.bgr.shape[:2], dtype=np.float32)
        with TemporaryDirectory() as temp:
            session = Path(temp)
            np.savez(session / "reference.npz", color_bgr=self.bgr, depth_m=depth,
                     metadata=json.dumps({"version": 1, "depth_delta_m": 0.025}))
            for i in range(6):
                np.savez(session / f"frame_{i:06d}.npz", color_bgr=self.bgr,
                         depth_m=depth * (0.5 if i < 4 else np.nan),
                         frame_id=i, color_timestamp_ms=i*200, depth_timestamp_ms=i*200)
            log = io.StringIO()
            with redirect_stdout(log):
                self.assertEqual(main(["replay", temp, "--no-display"]), 0)
            self.assertIn("Replayed 6 RGB frames: {'no_change': 6}", log.getvalue())

    def test_cli_record_and_offline_replay(self):
        # 30 discarded warm-up + one selection frame + 15 baseline + 6 scene frames.
        def packets():
            for i in range(52):
                color = self.bgr.copy()
                if 46 <= i < 50:
                    color[self.patch] = [90, 135, 190]
                yield FramePacket(color, i / 5, "desk_cam", i,
                                  serial_number="test", color_timestamp_ms=i*200.0)

        with TemporaryDirectory() as temp:
            session = Path(temp, "desk")
            record = ["record", str(session), "--serial", "test", "--no-display",
                      "--roi", "5", "4", "40", "32", "--max-frames", "6"]
            with patch("desk.RealSenseSource") as source, redirect_stdout(io.StringIO()):
                source.return_value.__enter__.return_value.frames.return_value = packets()
                self.assertEqual(main(record), 0)
                source.assert_called_once_with(serial="test", include_depth=False)
            with np.load(session / "reference.npz", allow_pickle=False) as ref:
                meta = json.loads(str(ref["metadata"]))
                self.assertEqual(meta["serial_number"], "test")
                self.assertEqual(meta["roi_xywh"], [5, 4, 40, 32])
                self.assertEqual(set(ref.files), {"color_bgr", "metadata"})
            self.assertEqual(len(list(session.glob("frame_*.npz"))), 6)
            with np.load(session / "frame_000000.npz", allow_pickle=False) as sample:
                self.assertEqual(sample["color_bgr"].shape, (32, 40, 3))
                self.assertEqual(set(sample.files), {"color_bgr", "frame_id", "color_timestamp_ms"})
                self.assertEqual(float(sample["color_timestamp_ms"]), 46*200)
            log = io.StringIO()
            with redirect_stdout(log), patch("desk.RealSenseSource") as source:
                self.assertEqual(main(["replay", str(session), "--no-display"]), 0)
                source.assert_not_called()
            self.assertIn("frame 48: object_candidate", log.getvalue())
            self.assertIn("frame 50: no_change", log.getvalue())
            self.assertIn("Replayed 6 RGB frames", log.getvalue())

    def gui_packets(self):
        for i in range(50):
            yield FramePacket(np.full_like(self.bgr, i), i / 5, "desk_cam", i,
                              serial_number="test", color_timestamp_ms=i*200.0)

    def test_live_freeze_reframe_and_explicit_record_key(self):
        # Cancel the first frozen selection, reframe from READY, then arm the second ROI.
        keys = iter([ord("r"), ord("f"), ord("f"), *([-1]*15), ord("b"),
                     -1, ord("F"), *([-1]*15), -1, ord("R"), ord("q")])
        rois = iter([(0, 0, 0, 0), (5, 4, 40, 32), (4, 4, 30, 30)])
        frozen_ids = []

        def select(title, image, **kwargs):
            frozen_ids.append(int(image[-1, -1, 0]))
            return next(rois)

        with TemporaryDirectory() as temp:
            session = Path(temp, "desk")

            def press_key(delay):
                key = next(keys)
                if key != ord("q"):  # All earlier key events, including R, precede the write.
                    self.assertFalse(session.exists())
                return key

            with (patch("desk.cv2.imshow"), patch("desk.cv2.waitKey", side_effect=press_key),
                  patch("desk.cv2.selectROI", side_effect=select), redirect_stdout(io.StringIO())):
                record_frames(self.gui_packets(), session, None, False, 10)
            self.assertEqual(frozen_ids, [1, 2, 20])
            with np.load(session / "reference.npz") as ref:
                self.assertEqual(json.loads(str(ref["metadata"]))["roi_xywh"], [4, 4, 30, 30])
                self.assertTrue(np.all(ref["color_bgr"] == 28))
            self.assertEqual(len(list(session.glob("frame_*.npz"))), 1)
            with np.load(session / "frame_000000.npz") as frame:
                self.assertEqual(int(frame["frame_id"]), 38)
                self.assertTrue(np.all(frame["color_bgr"] == 38))

    def test_quit_from_ready_writes_nothing(self):
        with TemporaryDirectory() as temp:
            session = Path(temp, "cancelled")
            keys = [ord("f"), *([-1]*15), 27]
            with (patch("desk.cv2.imshow"), patch("desk.cv2.waitKey", side_effect=keys),
                  redirect_stdout(io.StringIO())):
                record_frames(self.gui_packets(), session, (5, 4, 40, 32), False, 10)
            self.assertFalse(session.exists())

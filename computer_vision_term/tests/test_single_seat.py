import unittest

import numpy as np

from single_seat import Evidence, SeatConfirmation, SingleSeat


class SingleSeatTests(unittest.TestCase):
    def test_state_clock_runs_on_hold_and_restarts_only_on_a_confirmed_change(self):
        votes = SeatConfirmation()
        person = Evidence("occupied", "person", (0, 0, 20, 20), .9)
        item = Evidence("away", "RGB", (5, 5, 15, 15))
        self.assertIsNone(votes.update(0, [person])["state_elapsed_seconds"])
        for i in range(1, 25):
            state = votes.update(i / 4, [person])
        self.assertTrue(state["confirmed_this_frame"])
        self.assertEqual(state["state_elapsed_seconds"], 1)
        hold = votes.update(6.25, [])
        self.assertEqual((hold["state"], hold["state_elapsed_seconds"], hold["observed_seconds"]),
                         ("occupied", 1.25, 0))
        self.assertFalse(hold["confirmed_this_frame"])
        for i in range(26, 47):
            state = votes.update(i / 4, [item])
        self.assertEqual((state["state"], state["state_elapsed_seconds"]), ("away", 0))
        self.assertEqual(votes.update(12, [])["state_elapsed_seconds"], .5)
        self.assertIsNone(votes.update(0, [])["state_elapsed_seconds"])

    def test_dwell_fraction_and_duplicate_hits(self):
        item = Evidence("away", "book", (0, 0, 20, 20), .9)
        votes = SeatConfirmation()
        for i in range(50):
            s = votes.update(i/10, [] if i % 5 == 3 else [item, item])
            self.assertEqual(s["state"], "empty")
        s = votes.update(5, [item])
        self.assertEqual((s["state"], s["hits"], s["frames"]), ("away", 41, 51))

    def test_insufficient_or_unrelated_hits_and_camera_gap(self):
        a = Evidence("away", "RGB", (1, 1, 9, 9))
        b = Evidence("away", "book", (20, 20, 29, 29))
        votes = SeatConfirmation()
        for i in range(100):
            s = votes.update(i/10, [a if i % 2 else b])
            self.assertEqual(s["state"], "empty")
        s = votes.update(20, [b])
        self.assertEqual((s["observed_seconds"], s["frames"]), (0, 1))
        for i in range(1, 7):
            s = votes.update(20+i, [b])
        self.assertEqual(s["state"], "empty")  # Too few frames at 1 FPS.

    def test_known_unknown_chair_motion_person_priority_and_return(self):
        bg = np.full((120, 240, 3), 165, np.uint8)
        bg[40:100, 140:200] = 90
        def obj(name, box):
            return {"class_name": name, "confidence": .82, "bbox": box}
        chair = obj("chair", (140, 40, 200, 100))
        seat = SingleSeat(bg, (0, 15, 110, 100), (120, 20, 235, 110), [chair])
        i = 0
        def scene(image, detections):
            nonlocal i
            for _ in range(34):
                s = seat.observe(image, detections, i/5)
                i += 1
            return s["state"], s["source"]
        moved = bg.copy()
        moved[40:100, 140:200] = 165
        moved[40:100, 156:216] = 90
        chair = obj("chair", (156, 40, 216, 100))
        self.assertEqual(scene(moved, [chair]), ("empty", "clear"))
        bag = moved.copy()
        bag[55:76, 176:199] = 12
        self.assertEqual(scene(bag, [chair]), ("away", "RGB"))
        book = obj("book", (35, 25, 60, 45))
        self.assertEqual(scene(moved, [chair, book]), ("away", "book"))
        self.assertEqual(scene(bag, [obj("person", (120, 4, 225, 104)), book]), ("occupied", "person"))
        self.assertEqual(scene(moved, [chair, obj("person", (0, 2, 9, 13))]), ("empty", "clear"))

    def test_broad_occlusion_preserves_previous_state(self):
        bg = np.full((100, 200, 3), 170, np.uint8)
        s = SingleSeat(bg, (0, 0, 100, 100), (100, 0, 200, 100), [])
        view = bg.copy()
        view[:, 110:200] = (22, 40, 95)
        for i in range(41):
            result = s.observe(view, [], i/5)
        self.assertEqual((result["state"], result["candidate"]), ("empty", None))

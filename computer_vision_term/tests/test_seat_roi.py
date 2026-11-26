import unittest

from seat_roi import SeatROIManager


class SeatROITests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = SeatROIManager("seat_rois.json")

    @staticmethod
    def detection(bbox: list[int], class_name: str = "person") -> dict:
        return {
            "bbox": bbox,
            "class_name": class_name,
            "seat_candidates": [],
        }

    def test_uncalibrated_rois_do_not_make_candidates(self) -> None:
        detections = [self.detection([30, 70, 100, 200])]
        observations = self.manager.add_candidates(detections, (640, 480))
        self.assertEqual(observations, [])
        self.assertEqual(detections[0]["seat_candidates"], [])

    def test_detection_inside_one_seat(self) -> None:
        self.manager.calibrated = True
        detections = [self.detection([30, 70, 100, 200])]
        observations = self.manager.add_candidates(detections, (640, 480))

        self.assertEqual(detections[0]["seat_candidates"][0]["seat_id"], "1")
        self.assertEqual(detections[0]["seat_candidates"][0]["overlap_ratio"], 1.0)
        self.assertTrue(observations[0]["person_detected"])

    def test_detection_can_overlap_two_seats(self) -> None:
        self.manager.calibrated = True
        detections = [self.detection([100, 70, 150, 200], "backpack")]
        observations = self.manager.add_candidates(detections, (640, 480))
        candidate_ids = {
            candidate["seat_id"] for candidate in detections[0]["seat_candidates"]
        }

        self.assertEqual(candidate_ids, {"1", "2"})
        self.assertIn("backpack", observations[0]["belonging_classes"])
        self.assertIn("backpack", observations[1]["belonging_classes"])

    def test_detection_outside_all_seats(self) -> None:
        self.manager.calibrated = True
        detections = [self.detection([0, 0, 10, 10])]
        self.manager.add_candidates(detections, (640, 480))
        self.assertEqual(detections[0]["seat_candidates"], [])

    def test_empty_chair_is_localized_without_a_person_or_belonging(self) -> None:
        self.manager.calibrated = True
        detections = [self.detection([30, 70, 100, 200], "chair")]
        seat = self.manager.add_candidates(detections, (640, 480))[0]
        self.assertEqual(seat["detection_indices"], [0])
        self.assertFalse(seat["person_detected"])
        self.assertEqual(seat["belonging_classes"], [])


if __name__ == "__main__":
    unittest.main()

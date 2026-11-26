from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from camera import CameraIntrinsics, FramePacket
from stereo_board import detect_board, duplicate_pose, validate_calibration, write_report
from stereo_geometry import (board_points, camera_matrix, fit_stereo,
                             load_calibration, validate_transform)


class StereoTests(unittest.TestCase):
    serials = ("front", "rear")
    pattern = (9, 6)
    square_mm = 25
    intr = CameraIntrinsics(640, 480, 640., 635., 319., 238., "distortion.brown_conrady",
                            (-.06, .02, .001, -.001, 0.))
    intrinsics = dict.fromkeys(serials, intr)
    r12 = cv2.Rodrigues(np.array([.03, -.1, .04]))[0]
    t12 = np.array([-.19, .03, .02])

    @classmethod
    def pose(cls, i):
        board = board_points(cls.pattern, cls.square_mm)
        rotation = cv2.Rodrigues(np.array([.25*np.sin(i), .3*np.cos(i), .05*i]))[0]
        front = board @ rotation.T + [0.04*np.sin(i), -.08, .85 + .04*i]
        rear = front @ cls.r12.T + cls.t12
        return [(cv2.projectPoints(p, np.zeros(3), np.zeros(3), camera_matrix(cls.intr),
                                   np.array(cls.intr.coefficients))[0].astype(np.float32), p)
                for p in (front, rear)]

    @classmethod
    def setUpClass(cls):
        cls.samples = [tuple(uv for uv, xyz in cls.pose(i)) for i in range(10)]
        cls.fit = fit_stereo(cls.serials, cls.intrinsics, cls.samples, cls.pattern, cls.square_mm,
                             dict.fromkeys(cls.serials, 30), dict.fromkeys(cls.serials, .001))

    def test_detect_rendered_perspective_board(self):
        w, h = (self.pattern[0]+1)*40, (self.pattern[1]+1)*40
        y, x = np.indices((h, w))
        board = np.uint8(((x//40 + y//40) % 2)*255)
        quad = np.float32([[95, 81], [547, 65], [520, 378], [78, 349]])
        matrix = cv2.getPerspectiveTransform(np.float32([[0,0],[w,0],[w,h],[0,h]]), quad)
        img = cv2.warpPerspective(board, matrix, (640,480), borderValue=230)
        detected = detect_board(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), self.pattern)
        expected = cv2.perspectiveTransform((board_points(self.pattern, 1000)[:,None,:2]+1)*40, matrix)
        self.assertIsNotNone(detected)
        self.assertLess(min(np.linalg.norm(ordered-expected, axis=2).max()
                            for ordered in (detected,detected[::-1])), 2)
        self.assertIsNone(detect_board(np.zeros((200,300,3), np.uint8), self.pattern))

    def test_reverse_transform_and_epipolar_geometry_after_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"rig.json"
            self.fit.save(path)
            cal = load_calibration(path)
            with self.assertRaises(FileExistsError):
                self.fit.save(path)
        np.testing.assert_allclose(cal.cam1_from_cam2[:3,:3], self.r12.T, atol=1e-5)
        np.testing.assert_allclose(cal.cam1_from_cam2[:3,3], -self.r12.T@self.t12, atol=1e-5)
        (a, xyz), (b, other) = self.pose(11)
        result = cal.pixel_to_3d("rear", b[0,0], other[0,2], 42.)
        self.assertTrue(result["valid"])
        self.assertEqual(result["timestamp"], 42.)
        np.testing.assert_allclose(result["position_common_m"], xyz[0], atol=1e-5)
        self.assertLess(cal.epipolar_errors({"front":a, "rear":b}).max(), .001)
        self.assertGreater(cal.epipolar_errors({"front":a, "rear":b+[0.,15.]}).max(), 10.)

    def test_fit_from_two_rendered_checkerboard_streams(self):
        intr = CameraIntrinsics(640,480,600.,600.,320.,240.,"distortion.none",(0.,)*5)
        k = camera_matrix(intr)
        y, x = np.indices((280,400))
        board = np.uint8(((x//40+y//40) % 2)*255)
        image_edges = np.float32([[0,0],[400,0],[400,280],[0,280]])
        outer = np.float64([[-1,-1,0],[9,-1,0],[9,6,0],[-1,6,0]]) * .025
        def project(xyz):
            return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, None)[0].reshape(-1,2)
        samples = []
        for i in range(8):
            r = cv2.Rodrigues(np.array([.25*np.sin(i), .3*np.cos(i), .04*i]))[0]
            t = np.array([-.06+.02*np.sin(i), -.04, .63+.035*i])
            pair = []
            for dx in (0., -.085):
                position = t+[dx,0,0]
                vertices = np.float32(project(outer @ r.T + position))
                homography = cv2.getPerspectiveTransform(image_edges, vertices)
                gray = cv2.warpPerspective(board, homography, (640,480), borderValue=230)
                corners = detect_board(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), self.pattern)
                self.assertIsNotNone(corners)
                # The physical A mark plays this role in the operator's frozen preview.
                origin = project(position[None,:])[0]
                if np.linalg.norm(corners[-1,0]-origin) < np.linalg.norm(corners[0,0]-origin):
                    corners = corners[::-1].copy()
                pair.append(corners)
            samples.append(pair)
        fit = fit_stereo(self.serials, dict.fromkeys(self.serials,intr), samples, self.pattern,
                         self.square_mm, dict.fromkeys(self.serials,30), dict.fromkeys(self.serials,.001))
        np.testing.assert_allclose(fit.cam1_from_cam2[:3,3], [.085,0,0], atol=.003)

    def test_invalid_inputs_and_reversed_pair(self):
        for depth in [0., float("nan"), None]:
            self.assertFalse(self.fit.pixel_to_3d("rear", [100.,100.], depth, 0)["valid"])
        with self.assertRaises(ValueError):
            self.fit.transform_to_common_frame("wrong", [1.,2.,3.])
        with self.assertRaises(ValueError):
            self.fit.check_profile("front", self.intr, 60)
        with self.assertRaises(ValueError):
            validate_transform(np.diag([-1.,1.,1.,1.]))
        self.assertTrue(duplicate_pose(self.samples, tuple(a[::-1] for a in self.samples[0])))
        samples = list(self.samples)
        samples.append((samples[1][0], samples[1][1][::-1].copy()))
        fit = fit_stereo(self.serials, self.intrinsics, samples, self.pattern, self.square_mm,
                         self.fit.metadata['fps'], self.fit.metadata['depth_scale_m'])
        self.assertNotIn(len(samples), fit.metadata['accepted_pairs'])
        np.testing.assert_allclose(fit.cam1_from_cam2, self.fit.cam1_from_cam2, atol=1e-5)

    def test_independent_board_depth_and_report(self):
        packets, corners = [], []
        for s, (uv, xyz) in zip(self.serials, self.pose(12)):
            depth = np.zeros((480,640), np.float32)
            for q, p in zip(uv[1:,0], xyz[1:]):
                x,y = np.rint(q).astype(int)
                depth[y-2:y+3,x-2:x+3] = p[2]
            packets.append(FramePacket(np.zeros((480,640,3),np.uint8), 3., s, 5,
                                       serial_number=s, received_monotonic=3., depth_m=depth))
            corners.append(uv)
        rows = validate_calibration(self.fit, packets, corners, 1)
        self.assertFalse(rows[0]["valid"])
        self.assertTrue(all(r["valid"] and r["error_m"] < 1e-5 for r in rows[1:]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"validation.csv"
            write_report(path, rows)
            self.assertEqual(len(path.read_text().splitlines()), len(rows)+1)
            self.assertTrue(path.with_suffix(".png").is_file())

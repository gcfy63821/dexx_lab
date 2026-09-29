"""easy_handeye result -> camera-in-armbase 4x4 (tools/calib/handeye_to_extrinsic.py)."""
import importlib.util
import os
import tempfile
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HAVE_YAML = importlib.util.find_spec("yaml") is not None


def _tool():
    spec = importlib.util.spec_from_file_location(
        "handeye_to_extrinsic", os.path.join(REPO, "tools", "calib", "handeye_to_extrinsic.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CALIB2 = """parameters:
  name: test
  calibration_type: eye_on_base
  robot_base_frame: fr3_link0
  robot_effector_frame: fr3_link8
  tracking_base_frame: camera_color_optical_frame
  tracking_marker_frame: camera_marker
transform:
  translation: {x: 1.2, y: -0.1, z: 0.6}
  rotation: {x: 0.0, y: 0.0, z: 0.7071068, w: 0.7071068}
"""
CALIB1 = """eye_on_hand: false
robot_base_frame: fr3_link0
tracking_base_frame: camera_color_optical_frame
transformation: {x: 1.2, y: -0.1, z: 0.6, qx: 0.0, qy: 0.0, qz: 0.7071068, qw: 0.7071068}
"""


@unittest.skipUnless(HAVE_YAML, "PyYAML not installed")
class HandeyeToExtrinsicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _tool()

    def _write(self, d, text, name):
        p = os.path.join(d, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_both_formats_give_the_same_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            T2, m2 = self.t.load_handeye(self._write(d, CALIB2, "a.calib"))
            T1, m1 = self.t.load_handeye(self._write(d, CALIB1, "b.yaml"))
        np.testing.assert_allclose(T1, T2)
        np.testing.assert_allclose(T2[:3, 3], [1.2, -0.1, 0.6])
        np.testing.assert_allclose(T2[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
        self.assertEqual(m1["calibration_type"], "eye_on_base")

    def test_color_to_depth_composition(self):
        T_base_color = self.t.pose_to_matrix([1, 0, 0], [0, 0, 0, 1])
        color_T_depth = self.t.pose_to_matrix([0.05, 0, 0], [0, 0, 0, 1])
        np.testing.assert_allclose(self.t.convert(T_base_color, color_T_depth)[:3, 3], [1.05, 0, 0])

    def test_cli_writes_dated_file_and_refuses_current(self):
        with tempfile.TemporaryDirectory() as d:
            calib = self._write(d, CALIB2, "a.calib")
            out = os.path.join(d, "extrinsic_test.npy")
            self.assertEqual(self.t.main(["--calib", calib, "--out", out]), 0)
            T = np.load(out)
            self.assertEqual(T.shape, (4, 4))
            with self.assertRaises(SystemExit):
                self.t.main(["--calib", calib, "--out", os.path.join(d, "current.npy")])
            with self.assertRaises(SystemExit):  # exists, no --overwrite
                self.t.main(["--calib", calib, "--out", out])

    def test_cli_refuses_non_optical_tracking_frame(self):
        with tempfile.TemporaryDirectory() as d:
            calib = self._write(d, CALIB2.replace("camera_color_optical_frame", "camera_link"),
                                "body.calib")
            out = os.path.join(d, "extrinsic_body.npy")
            with self.assertRaises(SystemExit):
                self.t.main(["--calib", calib, "--out", out])
            self.assertFalse(os.path.exists(out))
            self.assertEqual(self.t.main(["--calib", calib, "--out", out, "--allow_non_optical"]), 0)
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()

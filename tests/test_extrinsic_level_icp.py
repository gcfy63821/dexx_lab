"""Pure-numpy pieces of the multi-pose ICP + table-levelling calibration tools.
No camera, robot, URDF or simulator."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "calib"))
HAVE_DEPS = importlib.util.find_spec("scipy") is not None


def rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def camera_looking_down(pos):
    """camera-in-armbase 4x4 (ROS optical) looking mostly down at the table."""
    # optical z forward = down-ish and toward +x, optical x = -y_world
    z = np.array([0.5, 0.0, -1.0]); z /= np.linalg.norm(z)
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = pos
    return T


@unittest.skipUnless(HAVE_DEPS, "needs scipy")
class ExtrinsicLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import calibrate_extrinsic_live_icp as icp_tool
        import level_extrinsic_to_table as level_tool
        cls.icp_tool, cls.level_tool = icp_tool, level_tool

    def test_backproject_principal_point_and_scale(self):
        intr = dict(fx=200.0, fy=200.0, cx=160.0, cy=120.0)
        depth = np.zeros((240, 320), np.float32)
        depth[120, 160] = 1.0          # principal point
        depth[120, 260] = 0.5          # 100 px right of it
        depth[0, 0] = 5.0              # beyond the far clamp -> dropped
        pts = self.icp_tool.backproject(depth, intr)
        self.assertEqual(len(pts), 2)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(pts[1], [0.25, 0.0, 0.5], atol=1e-6)

    def test_fit_plane_recovers_tilt_and_height(self):
        rng = np.random.default_rng(0)
        tz = self.icp_tool.TABLE_Z
        xy = rng.uniform([0.1, -0.4], [0.7, 0.25], size=(5000, 2))
        p = np.c_[xy, np.full(len(xy), tz)]
        piv = np.array([0.4, -0.1, tz])
        p = (p - piv) @ rot([1, 2, 0], 1.5).T + piv + [0, 0, 0.007]
        p[:200, 2] += 0.08                         # outliers (clutter on the table)
        n, dz = self.icp_tool.fit_plane(p, tz)
        self.assertAlmostEqual(self.icp_tool.tilt_deg(n), 1.5, places=2)
        self.assertAlmostEqual(dz, 0.007, delta=0.0005)

    def test_level_recovers_height_and_tilt_keeps_yaw(self):
        icp, lvl = self.icp_tool, self.level_tool
        tz = icp.TABLE_Z
        base = icp.ARM_BASE_POS.astype(float)
        T_true = camera_looking_down([0.1, 0.05, 0.75])

        # Table points seen by the true camera, expressed in the camera frame.
        rng = np.random.default_rng(1)
        xy = rng.uniform([0.1, -0.4], [0.7, 0.25], size=(8000, 2))
        tbl_env = np.c_[xy, np.full(len(xy), tz)]
        cam = (tbl_env - base - T_true[:3, 3]) @ T_true[:3, :3]

        # Initial guess: tilted 2 deg, 12 mm too high, and a 1 deg yaw error
        # that levelling must NOT touch.
        T_init = np.eye(4)
        dR = rot([0, 0, 1], 1.0) @ rot([1, -0.5, 0], 2.0)
        T_init[:3, :3] = dR @ T_true[:3, :3]
        T_init[:3, 3] = T_true[:3, 3] + [0.0, 0.0, 0.012]

        T_new, info = lvl.level_extrinsic(T_init, icp.cam_to_env(cam, T_init), tz)
        self.assertGreater(info["tilt_before_deg"], 1.5)

        after = icp.cam_to_env(cam, T_new)
        n, dz = icp.fit_plane(after, tz)
        self.assertLess(icp.tilt_deg(n), 1e-3)
        self.assertLess(abs(dz), 1e-5)
        np.testing.assert_allclose(after[:, 2], tz, atol=1e-5)
        # Levelling is a pure tilt about a horizontal axis: it maps the
        # initial rotation's yaw through unchanged, so the residual rotation
        # against the truth is the injected yaw only.
        R_res = T_new[:3, :3] @ T_true[:3, :3].T
        self.assertAlmostEqual(R_res[2, 2], 1.0, places=6)
        self.assertAlmostEqual(np.degrees(np.arctan2(R_res[1, 0], R_res[0, 0])), 1.0, delta=0.1)
        np.testing.assert_allclose(T_new[3], [0, 0, 0, 1])
        self.assertAlmostEqual(np.linalg.det(T_new[:3, :3]), 1.0, places=9)

    def test_level_is_identity_on_a_level_table(self):
        icp, lvl = self.icp_tool, self.level_tool
        tz = icp.TABLE_Z
        T = camera_looking_down([0.1, 0.0, 0.8])
        g = np.stack(np.meshgrid(np.linspace(0.1, 0.7, 60), np.linspace(-0.4, 0.2, 60)), -1)
        tbl = np.c_[g.reshape(-1, 2), np.full(3600, tz)]
        T_new, info = lvl.level_extrinsic(T, tbl, tz)
        np.testing.assert_allclose(T_new, T, atol=1e-9)
        self.assertLess(info["shift_mm"], 1e-6)

    def test_rotation_onto_z(self):
        for n in ([0, 0, 1], [0.02, -0.03, 1.0], [0.3, 0.1, 0.9]):
            R = self.level_tool.rotation_onto_z(n)
            v = R @ (np.asarray(n) / np.linalg.norm(n))
            np.testing.assert_allclose(v, [0, 0, 1], atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_cull_backfaces_and_voxel(self):
        pts = np.array([[0, 0, 0], [0, 0, 0]], float)
        nrm = np.array([[0, 0, 1], [0, 0, -1]], float)
        vis = self.icp_tool.cull_backfaces(pts, nrm, np.array([0, 0, 1.0]))
        self.assertEqual(vis.tolist(), [True, False])
        P = np.array([[0.001, 0, 0], [0.002, 0, 0], [0.02, 0, 0]])
        self.assertEqual(len(self.icp_tool.voxel_ds(P, 0.006)), 2)

    def test_output_path_guard(self):
        with self.assertRaises(SystemExit):
            self.icp_tool.check_out_path(str(ROOT / "calib/camera_align/current.npy"), True)


if __name__ == "__main__":
    unittest.main()

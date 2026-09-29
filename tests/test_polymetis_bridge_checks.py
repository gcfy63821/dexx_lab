"""Joint-target validation in deploy/polymetis_joint_bridge.py (polymetis stubbed)."""
import importlib.util
import os
import sys
import types
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_bridge():
    sys.modules.setdefault("polymetis", types.SimpleNamespace(RobotInterface=None))
    spec = importlib.util.spec_from_file_location(
        "polymetis_joint_bridge", os.path.join(REPO, "deploy", "polymetis_joint_bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CheckTargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.b = _load_bridge()
        except ImportError as exc:  # torch / zmq / msgpack missing
            raise unittest.SkipTest(str(exc))
        cls.home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], np.float32)

    def test_accepts_small_step(self):
        self.assertIsNone(self.b.check_target(self.home + 0.1, self.home, 0.5))

    def test_rejects_non_finite_and_wrong_size(self):
        q = self.home.copy(); q[2] = np.nan
        self.assertIn("finite", self.b.check_target(q, self.home, 0.5))
        self.assertIn("finite", self.b.check_target(self.home[:6], self.home, 0.5))

    def test_rejects_joint_limits(self):
        q = self.home.copy(); q[3] = 0.0  # joint 4 must stay below -0.1518
        self.assertIn("limits", self.b.check_target(q, self.home, 5.0))

    def test_rejects_large_step(self):
        self.assertIn("previous target", self.b.check_target(self.home + 0.6, self.home, 0.5))


if __name__ == "__main__":
    unittest.main()

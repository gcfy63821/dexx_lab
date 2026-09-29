"""Polymetis arm client against a local stand-in bridge; no robot or simulator."""
import importlib.util
import math
from pathlib import Path
import socket
import sys
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("torch", "zmq", "msgpack", "msgpack_numpy", "pytorch_kinematics"))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(HAVE_DEPS, "needs torch + pyzmq + msgpack + msgpack-numpy")
class PolymetisArmClientTests(unittest.TestCase):
    def setUp(self):
        from dexx.tasks.hand_imitation.deploy import polymetis_arm_client
        self.mod = polymetis_arm_client

    def test_angular_velocity_is_world_frame_and_shortest_path(self):
        import torch
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        a = 0.1  # rad about z in 0.1 s -> 1 rad/s
        q1 = torch.tensor([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        w = self.mod.world_angular_velocity(q0, q1, 0.1)
        torch.testing.assert_close(w, torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=0)
        # -q is the same rotation: the result must not jump by 2*pi / dt.
        w_neg = self.mod.world_angular_velocity(q0, -q1, 0.1)
        torch.testing.assert_close(w_neg, w, atol=1e-5, rtol=0)
        self.assertTrue(torch.equal(self.mod.world_angular_velocity(q1, q1, 0.1), torch.zeros(3)))

    def test_no_bridge_fails_at_startup(self):
        with self.assertRaisesRegex(RuntimeError, "no arm state"):
            self.mod.PolymetisArmClient("127.0.0.1", free_port(), free_port(),
                                        start_impedance=False, connect_timeout_s=0.5)

    def test_state_age_tracks_the_bridge(self):
        import torch
        import msgpack
        import msgpack_numpy
        import numpy as np
        import zmq
        msgpack_numpy.patch()
        state_port, cmd_port = free_port(), free_port()
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.bind(f"tcp://127.0.0.1:{state_port}")
        stop = threading.Event()

        def publish():
            while not stop.is_set():
                pub.send(msgpack.packb({
                    "joint_pos": np.arange(7, dtype=np.float32), "joint_vel": np.zeros(7, np.float32),
                    "ee_pos": np.zeros(3, np.float32),
                    "ee_quat_xyzw": np.array([0, 0, 0, 1], np.float32), "t": time.time()}))
                time.sleep(0.005)

        thread = threading.Thread(target=publish, daemon=True)
        thread.start()
        client = self.mod.PolymetisArmClient("127.0.0.1", state_port, cmd_port,
                                             start_impedance=False, connect_timeout_s=5.0)
        try:
            self.assertLess(client.state_age(), 0.1)
            self.assertEqual(client.arm_joint_positions.tolist(), list(range(7)))
            # The wrist is FK of the measured joints in sim's EE frame, not the
            # bridge's flange pose (published here as the origin).
            from dexx.tasks.hand_imitation.deploy.arm_fk import ArmFK
            pos, quat, v, w = ArmFK("right")(torch.arange(7, dtype=torch.float32), torch.zeros(7))
            got = client.wrist_state()
            torch.testing.assert_close(got[0], pos)
            torch.testing.assert_close(got[1], quat)
            self.assertEqual(float(got[2].abs().sum() + got[3].abs().sum()), 0.0)
            self.assertEqual(client.flange_position.tolist(), [0.0, 0.0, 0.0])
            stop.set()
            thread.join()
            time.sleep(0.3)
            self.assertGreater(client.state_age(), 0.25)
        finally:
            stop.set()
            client.shutdown(terminate_policy=False)
            pub.close(0)


if __name__ == "__main__":
    unittest.main()

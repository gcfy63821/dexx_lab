"""ROS2 arm backend: wrist publisher math, subscriber callbacks, adapter guards.

Needs rclpy for the subscriber/adapter tests (skipped otherwise); the wrist
publisher's FK and finite-difference helpers are checked on their own.
"""
import importlib.util
import os
import time
import types
import unittest

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    import rclpy  # noqa: F401
    from sensor_msgs.msg import JointState  # noqa: F401
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False

try:
    import pytorch_kinematics as pk
    HAVE_PK = True
except ImportError:
    HAVE_PK = False


def _load_wrist_publisher():
    spec = importlib.util.spec_from_file_location(
        "wrist_state_publisher", os.path.join(REPO, "deploy", "ros2", "wrist_state_publisher.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAVE_ROS and HAVE_PK, "rclpy / pytorch_kinematics not available")
class WristPublisherTest(unittest.TestCase):
    """The publisher reports exactly ArmFK's wrist state, and goes quiet when stale."""

    @classmethod
    def setUpClass(cls):
        cls.wp = _load_wrist_publisher()
        if not rclpy.ok():
            rclpy.init()
        cls.node = cls.wp.WristStatePublisher(
            side="right", rate=200.0, body_name="right_hand_C_MC", joint_states_topic="/test_js",
            output_topic="/test_wrist", repo_root=REPO, max_age_s=0.1)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()

    def _msg(self, q, dq, sec=1, nanosec=0):
        names = ["fr3_finger_joint1"] + [f"fr3_joint{i}" for i in range(1, 8)]
        stamp = types.SimpleNamespace(sec=sec, nanosec=nanosec)
        return types.SimpleNamespace(name=names, position=[0.0] + list(q), velocity=[0.0] + list(dq),
                                     header=types.SimpleNamespace(stamp=stamp))

    def test_state_matches_armfk_and_goes_quiet_when_stale(self):
        from dexx.tasks.hand_imitation.deploy.arm_fk import ArmFK
        q = [0.1, -0.5, 0.2, -2.2, 0.3, 1.8, 0.5]
        dq = [0.2, -0.1, 0.05, 0.3, -0.2, 0.1, 0.4]
        self.node._joint_state_cb(self._msg(q, dq))
        ref = ArmFK("right")(torch.tensor(q), torch.tensor(dq))
        for got, want in zip(self.node._state, ref):
            np.testing.assert_allclose(got, want.numpy(), atol=1e-6)
        sent = []
        self.node._pub = types.SimpleNamespace(publish=sent.append)
        self.node._tick()
        self.assertEqual(len(sent), 1)
        self.node._last_rx -= 1.0  # joint states stopped a second ago
        self.node._tick()
        self.assertEqual(len(sent), 1)


@unittest.skipUnless(HAVE_ROS, "rclpy not available")
class SubscriberAndAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dexx.tasks.hand_imitation.deploy import ros2_arm_client, ros2_observation_subscriber
        cls.rac, cls.sub_mod = ros2_arm_client, ros2_observation_subscriber
        cls.owns_rclpy = not rclpy.ok()
        if cls.owns_rclpy:
            rclpy.init()
        cls.sub = ros2_observation_subscriber.ROS2ObservationSubscriber(namespace="")

    @classmethod
    def tearDownClass(cls):
        cls.sub.destroy_node()
        if cls.owns_rclpy:
            rclpy.shutdown()

    def test_joint_states_by_name_and_wrist_state(self):
        names = ["fr3_finger_joint1"] + [f"fr3_joint{i}" for i in (3, 1, 2, 7, 4, 5, 6)]
        pos = [9.0] + [0.1 * int(n[-1]) for n in names[1:]]
        self.sub.joint_state_callback(types.SimpleNamespace(name=names, position=pos, velocity=[]))
        torch.testing.assert_close(self.sub.arm_joint_positions, torch.arange(1, 8) * 0.1)
        self.assertIsNotNone(self.sub.last_joint_state_rx)
        data = [1, 2, 3, 0.0, 0.0, 0.7071068, 0.7071068, 4, 5, 6, 7, 8, 9]
        self.sub.wrist_state_callback(types.SimpleNamespace(data=data))
        torch.testing.assert_close(self.sub.wrist_quaternion,
                                   torch.tensor([0.7071068, 0.0, 0.0, 0.7071068]))
        torch.testing.assert_close(self.sub.wrist_angular_velocity, torch.tensor([7.0, 8.0, 9.0]))
        self.assertIs(self.sub.wrist_state_snapshot[1], self.sub.wrist_quaternion)
        self.assertTrue(self.sub.wrist_state_topic_active)

    def _adapter(self):
        c = self.rac.Ros2ArmClient.__new__(self.rac.Ros2ArmClient)  # skip the connect wait
        c.logger = self.rac._make_logger("test")
        c.obs = types.SimpleNamespace(last_joint_state_rx=None, last_wrist_state_rx=None)
        c.pub = types.SimpleNamespace(sent=[], publish_arm_joint_pos=lambda q: c.pub.sent.append(q))
        return c

    def test_state_age_is_the_older_stream(self):
        c = self._adapter()
        self.assertEqual(c.state_age(), float("inf"))
        now = time.monotonic()
        c.obs.last_joint_state_rx, c.obs.last_wrist_state_rx = now, now - 1.0
        self.assertGreaterEqual(c.state_age(), 1.0)

    def test_non_finite_target_is_dropped(self):
        c = self._adapter()
        c.publish_arm_joint_pos(torch.tensor([0.0, float("nan"), 0, 0, 0, 0, 0]))
        c.publish_arm_joint_pos(np.zeros(6))
        self.assertEqual(c.pub.sent, [])
        c.publish_arm_joint_pos(torch.zeros(1, 7))
        self.assertEqual(len(c.pub.sent), 1)


if __name__ == "__main__":
    unittest.main()

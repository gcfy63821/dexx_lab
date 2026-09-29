"""ROS2 arm backend for the Sharpa deploy env (optional, experimental).

Wraps the research ROS2 comm layer — ``ROS2ObservationSubscriber`` (arm and
wrist state) and ``ROS2ActionPublisher`` (joint targets on
``/teleop_joint_commands``) spun on one background executor — behind the same
interface as ``PolymetisArmClient``, so the deploy env does not know which
backend it talks to.

    robot PC                                   inference PC
    DexhandJointImpedanceController  <--- /teleop_joint_commands  ROS2ActionPublisher
    joint_state_broadcaster  --- /joint_states --->  ROS2ObservationSubscriber
                                         \\--->  wrist_state_publisher.py
                                                 --- /franka_wrist_state (200 Hz) --->

The wrist comes from ``deploy/ros2/wrist_state_publisher.py`` (FK of the
measured joints on the sim's URDF at ``right_hand_C_MC``, finite-differenced
velocities); it must be running. The PoseStamped fallback of the subscriber is
refused here: it has no velocities and is a different body.

The impedance gains live in the controller's YAML on the robot PC, so
``start_joint_impedance`` / ``go_home`` are no-ops, and on shutdown the
controller keeps holding the last target.

Interface exposed (what the deploy env reads/calls):
    read attrs : arm_joint_positions, arm_joint_velocities,
                 wrist_position, wrist_quaternion (w,x,y,z),
                 wrist_linear_velocity, wrist_angular_velocity,
                 arm_data_received, wrist_msg_count
    methods    : publish_arm_joint_pos(q7), state_age(), get_logger(), shutdown()
"""

from __future__ import annotations

import logging
import threading
import time


def _make_logger(name: str = "Ros2ArmClient") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    if not hasattr(logger, "warn"):
        logger.warn = logger.warning  # type: ignore[attr-defined]
    return logger


class Ros2ArmClient:
    """ROS2ObservationSubscriber + ROS2ActionPublisher with the deploy-env interface."""

    def __init__(self, namespace: str = "", connect_timeout_s: float = 10.0,
                 logger: logging.Logger | None = None):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from dexx.tasks.hand_imitation.deploy.ros2_action_publisher import ROS2ActionPublisher
            from dexx.tasks.hand_imitation.deploy.ros2_observation_subscriber import (
                ROS2ObservationSubscriber,
            )
        except ImportError as exc:
            raise ImportError(
                "the ROS2 arm backend needs rclpy, sensor_msgs, geometry_msgs and std_msgs: "
                "`source /opt/ros/humble/setup.bash` before running.\n  %r" % (exc,)
            )
        self.logger = logger or _make_logger()
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self._rclpy = rclpy
        self.obs = ROS2ObservationSubscriber(namespace=namespace)
        self.pub = ROS2ActionPublisher(namespace=namespace)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.obs)
        self._executor.add_node(self.pub)
        self._stop = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        ns = namespace.rstrip("/")
        self.logger.info(f"waiting for {ns}/joint_states and {ns}/franka_wrist_state ...")
        t0 = time.time()
        while time.time() - t0 < connect_timeout_s:
            if self.obs.arm_data_received and self.obs.wrist_state_topic_active:
                break
            time.sleep(0.05)
        missing = []
        if not self.obs.arm_data_received:
            missing.append(f"{ns}/joint_states (is the robot-side controller running, same ROS_DOMAIN_ID?)")
        if not self.obs.wrist_state_topic_active:
            missing.append(f"{ns}/franka_wrist_state (start deploy/ros2/wrist_state_publisher.py)")
        if missing:
            self.shutdown()
            raise RuntimeError(f"no data within {connect_timeout_s}s on: " + "; ".join(missing))

    def _spin(self):
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    self.logger.warning(f"executor error: {exc!r}")

    # ---- state (read through to the subscriber) ----
    arm_joint_positions = property(lambda self: self.obs.arm_joint_positions)
    arm_joint_velocities = property(lambda self: self.obs.arm_joint_velocities)
    wrist_position = property(lambda self: self.obs.wrist_position)
    wrist_quaternion = property(lambda self: self.obs.wrist_quaternion)
    wrist_linear_velocity = property(lambda self: self.obs.wrist_linear_velocity)
    wrist_angular_velocity = property(lambda self: self.obs.wrist_angular_velocity)
    arm_data_received = property(lambda self: self.obs.arm_data_received)
    wrist_msg_count = property(lambda self: self.obs.wrist_msg_count)

    def wrist_state(self):
        """(pos, quat wxyz, lin vel, ang vel) from the newest /franka_wrist_state, one message."""
        return self.obs.wrist_state_snapshot

    def state_age(self) -> float:
        """Seconds since the older of the newest /joint_states and /franka_wrist_state
        (inf before both arrived): either one going stale makes the obs stale."""
        a, w = self.obs.last_joint_state_rx, self.obs.last_wrist_state_rx
        if a is None or w is None:
            return float("inf")
        return time.monotonic() - min(a, w)

    # ---- commands ----
    def publish_arm_joint_pos(self, arm_joint_pos_des):
        import numpy as np
        import torch
        q = (arm_joint_pos_des.detach().cpu().flatten().float().numpy()
             if isinstance(arm_joint_pos_des, torch.Tensor)
             else np.asarray(arm_joint_pos_des, dtype=np.float32).flatten())
        # The publisher would replace NaN/inf with 0 rad — a real joint target.
        # Drop the command instead, as PolymetisArmClient does.
        if q.size != 7 or not np.isfinite(q).all():
            self.logger.warning(f"invalid arm target (size {q.size}, finite={np.isfinite(q).all()}); ignoring")
            return
        self.pub.publish_arm_joint_pos(q)

    def start_joint_impedance(self):
        self.logger.info("ROS2 backend: impedance gains are set in the robot-side controller YAML")

    def go_home(self):
        self.logger.warning("ROS2 backend: go_home is not supported; move the arm with joint targets")

    def get_logger(self):
        return self.logger

    def shutdown(self, terminate_policy: bool = True):
        """Stop spinning and destroy the nodes. The controller keeps holding the
        last target (``terminate_policy`` is accepted for interface parity)."""
        self._stop.set()
        try:
            if self._spin_thread.is_alive():
                self._spin_thread.join(timeout=1.0)
        except Exception:
            pass
        for node in (getattr(self, "obs", None), getattr(self, "pub", None)):
            try:
                self._executor.remove_node(node)
                node.destroy_node()
            except Exception:
                pass
        if self._owns_rclpy:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass

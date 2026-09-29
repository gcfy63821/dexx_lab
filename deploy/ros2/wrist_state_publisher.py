#!/usr/bin/env python3
"""ROS2 publisher: /franka_wrist_state Float32MultiArray (13d).

Runs on the inference PC (ROS2 backend), before deploy/deploy_pc.py:

    python deploy/ros2/wrist_state_publisher.py --side right --rate 200
    ros2 topic hz /franka_wrist_state          # ~200 Hz

Subscribes /joint_states and computes the Sharpa hand-base body
(right_hand_C_MC / left_hand_C_MC) with dexx's ArmFK on the sim's URDF
(assets/generated/fr3_with_{side}_sharpa_wave.urdf): position, the quaternion
with PhysX's sign, centre-of-mass linear velocity and angular velocity from
J(q)·dq — the same quantities the simulator reports. Publishes at --rate Hz,
and stops publishing when /joint_states goes quiet, so the deploy's staleness
check sees it.

Why: the deploy env reads wrist obs from either:
  (a) /franka_wrist_state Float32MultiArray  ← preferred, has velocities
  (b) /franka_robot_state_broadcaster/current_pose  ← PoseStamped fallback,
      publishes fr3_hand body — different from sim's right_hand_C_MC,
      causes ~1.7 deg quat residual that is config-dependent (cannot be
      fixed by a single cfg.wrist_quat_offset).

This publisher emits (a) using the same URDF and same body name sim uses,
so the sim2real wrist-quat gap collapses to whatever steady-state error
the arm impedance produces (~few mrad).

Format (matches src/dexx/tasks/hand_imitation/deploy/ros2_observation_subscriber.py):
    data[0:3]   pos x, y, z
    data[3:7]   quat x, y, z, w   (subscriber converts to wxyz internally)
    data[7:10]  lin_vel x, y, z
    data[10:13] ang_vel x, y, z
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


ARM_JOINT_NAMES = [
    'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
    'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
]


class WristStatePublisher(Node):
    def __init__(
        self,
        side: str,
        rate: float,
        body_name: str,
        joint_states_topic: str,
        output_topic: str,
        repo_root: str,
        log_period_s: float = 5.0,
        max_age_s: float = 0.1,
    ):
        super().__init__('wrist_state_publisher')
        self._rate = rate
        self._body = body_name

        urdf_filename = f'fr3_with_{side}_sharpa_wave.urdf'
        urdf_path = os.path.join(repo_root, 'assets', 'generated', urdf_filename)
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f'URDF not found: {urdf_path}')
        sys.path.insert(0, os.path.join(repo_root, 'src'))
        from dexx.tasks.hand_imitation.deploy.arm_fk import ArmFK
        self._fk = ArmFK(side=side, urdf_path=urdf_path, ee_link=body_name)
        self._max_age = max_age_s

        # state
        self._arm_q = np.zeros(7, dtype=np.float32)
        self._prev_q = None
        self._prev_t = None
        self._last_rx = None
        self._state = None  # (pos, quat_wxyz, lin_vel, ang_vel) of the newest reading
        self._n_skip_stale = 0

        # ROS plumbing
        self.create_subscription(JointState, joint_states_topic, self._joint_state_cb, 10)
        self._pub = self.create_publisher(Float32MultiArray, output_topic, 10)
        self._timer = self.create_timer(1.0 / rate, self._tick)

        self._n_pub = 0
        self._n_skip_no_q = 0
        self._last_log_t = time.monotonic()
        self._log_period = log_period_s

        self.get_logger().info(f'[wrist_pub] URDF      : {urdf_path}')
        self.get_logger().info(f'[wrist_pub] body      : {body_name}')
        self.get_logger().info(f'[wrist_pub] subscribe : {joint_states_topic}')
        self.get_logger().info(f'[wrist_pub] publish   : {output_topic} @ {rate:.0f} Hz')
        self.get_logger().info(
            '[wrist_pub] format: [pos(3), quat_xyzw(4), lin_vel(3), ang_vel(3)]'
        )

    def _joint_state_cb(self, msg: JointState):
        """Wrist state from every new joint reading. Joint velocities come from
        the message; if it carries none, they are differenced over the stamps."""
        try:
            if not all(jn in msg.name for jn in ARM_JOINT_NAMES):
                return
            idx = [msg.name.index(jn) for jn in ARM_JOINT_NAMES]
            q = np.array([msg.position[i] for i in idx], dtype=np.float32)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            t = stamp if stamp > 0 else time.monotonic()
            if len(msg.velocity) == len(msg.name):
                dq = np.array([msg.velocity[i] for i in idx], dtype=np.float32)
            elif self._prev_t is not None and t > self._prev_t:
                dq = (q - self._prev_q) / (t - self._prev_t)
            else:
                dq = np.zeros(7, dtype=np.float32)
            pos, quat, v, w = self._fk(torch.from_numpy(q), torch.from_numpy(dq))
            self._state = (pos.numpy(), quat.numpy(), v.numpy(), w.numpy())
            self._prev_q, self._prev_t = q, t
            self._last_rx = time.monotonic()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'joint_state cb: {e}')

    def _tick(self):
        """Publish the newest state at --rate, only while /joint_states is fresh."""
        if self._state is None:
            self._n_skip_no_q += 1
            self._maybe_log_status()
            return
        if time.monotonic() - self._last_rx > self._max_age:
            self._n_skip_stale += 1
            self._maybe_log_status()
            return
        pos, quat_wxyz, lin_vel, ang_vel = self._state

        # subscriber expects xyzw at indices 3..7 (then converts to wxyz)
        out = Float32MultiArray()
        out.data = [
            float(pos[0]), float(pos[1]), float(pos[2]),
            float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]),
            float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2]),
            float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2]),
        ]
        self._pub.publish(out)
        self._n_pub += 1
        self._maybe_log_status(pos, lin_vel)

    def _maybe_log_status(self, pos=None, lin_vel=None):
        now = time.monotonic()
        if now - self._last_log_t < self._log_period:
            return
        elapsed = now - self._last_log_t
        rate = self._n_pub / max(elapsed, 1e-6)
        if self._n_pub == 0:
            self.get_logger().warn(
                f'[wrist_pub] 0 published in {elapsed:.1f}s '
                f'(no /joint_states: {self._n_skip_no_q} ticks; stale /joint_states: '
                f'{self._n_skip_stale} ticks)'
            )
        else:
            extra = ''
            if pos is not None:
                extra = f' pos={pos.tolist()} |v|={np.linalg.norm(lin_vel):.3f}'
            self.get_logger().info(
                f'[wrist_pub] pub_rate={rate:.1f}Hz over {elapsed:.1f}s{extra}'
            )
        self._n_pub = 0
        self._n_skip_no_q = 0
        self._n_skip_stale = 0
        self._last_log_t = now


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side', choices=['right', 'left'], default='right')
    p.add_argument('--rate', type=float, default=200.0,
                   help='Publish rate in Hz (default 200).')
    p.add_argument('--namespace', type=str, default='',
                   help='ROS2 namespace prefix (default empty, matches deploy env).')
    p.add_argument('--body', type=str, default=None,
                   help='URDF body to publish; default right_hand_C_MC / left_hand_C_MC.')
    p.add_argument('--joint_states_topic', type=str, default=None,
                   help='Override /joint_states topic; default <ns>/joint_states.')
    p.add_argument('--output_topic', type=str, default=None,
                   help='Override output topic; default <ns>/franka_wrist_state.')
    p.add_argument('--max_age', type=float, default=0.1,
                   help='Stop publishing when the newest /joint_states is older than this (s).')
    p.add_argument('--repo_root', type=str,
                   default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                   help='Repo root used to locate assets/generated/*.urdf.')
    args = p.parse_args()

    body = args.body or ('right_hand_C_MC' if args.side == 'right' else 'left_hand_C_MC')
    ns = args.namespace.rstrip('/')
    js_topic = args.joint_states_topic or (f'{ns}/joint_states' if ns else '/joint_states')
    out_topic = args.output_topic or (f'{ns}/franka_wrist_state' if ns else '/franka_wrist_state')

    rclpy.init()
    try:
        node = WristStatePublisher(
            side=args.side,
            rate=args.rate,
            body_name=body,
            joint_states_topic=js_topic,
            output_topic=out_topic,
            repo_root=args.repo_root,
            max_age_s=args.max_age,
        )
    except Exception as e:
        print(f'[wrist_pub] init failed: {e}', file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 node to subscribe to Franka arm observations.

Topics (``namespace`` prefix, empty on the reference setup):
    /joint_states                                  sensor_msgs/JointState  (fr3_joint1..7 by name)
    /franka_wrist_state                            Float32MultiArray (13): pos, quat xyzw, lin vel, ang vel
                                                   from deploy/ros2/wrist_state_publisher.py
    /franka_robot_state_broadcaster/current_pose   PoseStamped fallback (no velocities, fr3_hand body)
"""

import time

import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


class ROS2ObservationSubscriber(Node):
    """ROS2 node to subscribe to Franka arm observations."""
    
    ARM_JOINT_NAMES = [
        'fr3_joint1',
        'fr3_joint2',
        'fr3_joint3',
        'fr3_joint4',
        'fr3_joint5',
        'fr3_joint6',
        'fr3_joint7'
    ]
    
    def __init__(self, namespace: str = '/NS_1'):
        super().__init__('franka_observation_subscriber')
        
        self.namespace = namespace.rstrip('/')  # Remove trailing slash if present
        
        # Data storage
        self.arm_joint_positions = torch.zeros(7, dtype=torch.float32)
        self.arm_joint_velocities = torch.zeros(7, dtype=torch.float32)
        self.wrist_position = torch.zeros(3, dtype=torch.float32)
        self.wrist_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)  # [w, x, y, z]
        self.wrist_linear_velocity = torch.zeros(3, dtype=torch.float32)
        self.wrist_angular_velocity = torch.zeros(3, dtype=torch.float32)
        
        # Flags
        self.arm_data_received = False
        self.wrist_data_received = False
        # Distinguishes which wrist source is live. If Float32MultiArray /franka_wrist_state
        # is being published, prefer it (has velocities). Otherwise fall back to PoseStamped
        # and keep updating it every message.
        self.wrist_state_topic_active = False
        self.wrist_msg_count = 0
        # Local receive times (time.monotonic) for the deploy env's stale-sensor check.
        self.last_joint_state_rx = None
        self.last_wrist_state_rx = None
        self.wrist_state_snapshot = None
        
        # Build topic names with namespace
        joint_states_topic = f'{self.namespace}/joint_states'
        wrist_state_topic = f'{self.namespace}/franka_wrist_state'  # from deploy/ros2/wrist_state_publisher.py
        wrist_pose_topic = f'{self.namespace}/franka_robot_state_broadcaster/current_pose'
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            joint_states_topic,
            self.joint_state_callback,
            10
        )
        
        # Wrist state from deploy/ros2/wrist_state_publisher.py (preferred: has velocities)
        self.wrist_state_sub = self.create_subscription(
            Float32MultiArray,
            wrist_state_topic,
            self.wrist_state_callback,
            10
        )
        
        # Fallback: subscribe to wrist pose directly
        self.wrist_pose_sub = self.create_subscription(
            PoseStamped,
            wrist_pose_topic,
            self.wrist_pose_callback,
            10
        )
        
        self.get_logger().info('Franka observation subscriber initialized')
        self.get_logger().info(f'  Namespace: {self.namespace}')
        self.get_logger().info('  Subscribed to:')
        self.get_logger().info(f'    - {joint_states_topic}')
        self.get_logger().info(f'    - {wrist_state_topic} (optional)')
        self.get_logger().info(f'    - {wrist_pose_topic}')
    
    def joint_state_callback(self, msg):
        """Callback for joint state messages."""
        try:
            # Extract arm joint positions
            arm_positions = []
            arm_velocities = []
            
            for joint_name in self.ARM_JOINT_NAMES:
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    arm_positions.append(msg.position[idx])
                    if len(msg.velocity) > idx:
                        arm_velocities.append(msg.velocity[idx])
                    else:
                        arm_velocities.append(0.0)
            
            if len(arm_positions) == 7:
                self.arm_joint_positions = torch.tensor(arm_positions, dtype=torch.float32)
                self.arm_joint_velocities = torch.tensor(arm_velocities, dtype=torch.float32)
                self.arm_data_received = True
                self.last_joint_state_rx = time.monotonic()
        except Exception as e:
            self.get_logger().error(f'Error processing joint states: {e}')
    
    def wrist_state_callback(self, msg):
        """Callback for wrist state (Float32MultiArray with 13 elements)."""
        try:
            if len(msg.data) >= 13:
                # Format: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
                self.wrist_position = torch.tensor(msg.data[0:3], dtype=torch.float32)
                # Quaternion in msg: [x, y, z, w], convert to [w, x, y, z]
                quat_xyzw = msg.data[3:7]
                self.wrist_quaternion = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32)
                self.wrist_linear_velocity = torch.tensor(msg.data[7:10], dtype=torch.float32)
                self.wrist_angular_velocity = torch.tensor(msg.data[10:13], dtype=torch.float32)
                # One snapshot of all four, swapped in atomically for readers on other threads.
                self.wrist_state_snapshot = (self.wrist_position, self.wrist_quaternion,
                                             self.wrist_linear_velocity, self.wrist_angular_velocity)
                self.wrist_state_topic_active = True
                self.wrist_data_received = True
                self.wrist_msg_count += 1
                self.last_wrist_state_rx = time.monotonic()
        except Exception as e:
            self.get_logger().error(f'Error processing wrist state: {e}')

    def wrist_pose_callback(self, msg):
        """Fallback callback for wrist pose (PoseStamped). Updates every message
        unless Float32MultiArray /franka_wrist_state is also being published
        (in which case we prefer that source for its velocities)."""
        try:
            if self.wrist_state_topic_active:
                return  # prefer Float32MultiArray source
            self.wrist_position = torch.tensor([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ], dtype=torch.float32)

            # ROS quaternion: [x, y, z, w] -> [w, x, y, z]
            self.wrist_quaternion = torch.tensor([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ], dtype=torch.float32)

            # Velocities not available from PoseStamped, keep zeros (finite-diff
            # is done by the env from base_pos/base_quat history).
            self.wrist_data_received = True
            self.wrist_msg_count += 1
        except Exception as e:
            self.get_logger().error(f'Error processing wrist pose: {e}')


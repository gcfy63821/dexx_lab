# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS2 node to publish actions to Franka arm."""

import torch
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
import math


class ROS2ActionPublisher(Node):
    """ROS2 node to publish actions to Franka arm."""
    
    HAND_JOINT_NAMES = [
        "left_thumb_CMC_FE",
        "left_thumb_CMC_AA",
        "left_thumb_MCP_FE",
        "left_thumb_MCP_AA",
        "left_thumb_IP",
        "left_index_MCP_FE",
        "left_index_MCP_AA",
        "left_index_PIP",
        "left_index_DIP",
        "left_middle_MCP_FE",
        "left_middle_MCP_AA",
        "left_middle_PIP",
        "left_middle_DIP",
        "left_ring_MCP_FE",
        "left_ring_MCP_AA",
        "left_ring_PIP",
        "left_ring_DIP",
        "left_pinky_CMC",
        "left_pinky_MCP_FE",
        "left_pinky_MCP_AA",
        "left_pinky_PIP",
        "left_pinky_DIP",
    ]
    
    ARM_JOINT_NAMES = [
        'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
        'fr3_joint5', 'fr3_joint6', 'fr3_joint7',
    ]

    def __init__(self, namespace: str = ''):
        super().__init__('franka_action_publisher')
        # Same prefix as the state topics, so a namespaced robot is commanded
        # on its own controller's topic.
        self.command_topic = f"{namespace.rstrip('/')}/teleop_joint_commands"

        # BestEffort QoS to match teleop controller subscription
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        qos_best_effort = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers
        self.hand_joint_pub = self.create_publisher(
            JointState,
            'hand_joint_positions',
            10
        )

        self.wrist_pose_pub = self.create_publisher(
            PoseStamped,
            'wrist_pose',
            10
        )

        # Arm joint command: JointState on /teleop_joint_commands (matches TeleopJointImpedanceController)
        self.arm_joint_pos_pub = self.create_publisher(
            JointState,
            self.command_topic,
            qos_best_effort,
        )

        # Legacy Float32MultiArray publisher (kept for compatibility)
        self.arm_joint_pos_legacy_pub = self.create_publisher(
            Float32MultiArray,
            'arm_joint_positions',
            10
        )

        self.get_logger().info('Franka action publisher initialized')
        self.get_logger().info('  Publishing to:')
        self.get_logger().info(f'    - {self.command_topic} (sensor_msgs/JointState, BestEffort)')
        self.get_logger().info('    - hand_joint_positions (sensor_msgs/JointState)')
        self.get_logger().info('    - wrist_pose (geometry_msgs/PoseStamped)')
    
    def publish_hand_joints(self, joint_positions):
        """Publish hand joint positions.
        
        Args:
            joint_positions: Tensor, numpy array, or list of joint positions
        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "hand"
        
        # Convert to numpy array first
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()
        
        # Flatten if 2D (e.g., [1, 22] -> [22])
        joint_positions = joint_positions.flatten()
        
        # Convert to list of Python floats and ensure valid values
        import math
        position_list = []
        for x in joint_positions:
            val = float(x)
            # Check for NaN or inf
            if math.isnan(val) or math.isinf(val):
                self.get_logger().warn(f'Invalid joint position detected: {val}, replacing with 0.0')
                val = 0.0
            position_list.append(val)
        
        msg.name = self.HAND_JOINT_NAMES[:len(position_list)]
        msg.position = position_list
        msg.velocity = []
        msg.effort = []
        self.hand_joint_pub.publish(msg)
    
    def publish_wrist_pose(self, position, quaternion):
        """Publish wrist pose (position and quaternion).
        
        Args:
            position: [x, y, z] tensor or array
            quaternion: [w, x, y, z] tensor or array
        """
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        
        # Convert to numpy if tensor
        if isinstance(position, torch.Tensor):
            position = position.cpu().numpy()
        if isinstance(quaternion, torch.Tensor):
            quaternion = quaternion.cpu().numpy()
        
        # Set position
        msg.pose.position = Point()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        
        # Set orientation (quaternion: [w, x, y, z] -> ROS [x, y, z, w])
        msg.pose.orientation = Quaternion()
        msg.pose.orientation.w = float(quaternion[0])  # w component
        msg.pose.orientation.x = float(quaternion[1])  # x component
        msg.pose.orientation.y = float(quaternion[2])  # y component
        msg.pose.orientation.z = float(quaternion[3])  # z component
        
        self.wrist_pose_pub.publish(msg)

    def publish_arm_joint_pos(self, arm_joint_pos_des):
        """Publish arm joint target positions for Franka arm.

        Publishes as sensor_msgs/JointState on /teleop_joint_commands
        (matching TeleopJointImpedanceController subscription).

        Args:
            arm_joint_pos_des: Tensor, numpy array, or list of 7 joint positions (rad)
        """
        # Convert to numpy if tensor
        if isinstance(arm_joint_pos_des, torch.Tensor):
            arm_joint_pos_des = arm_joint_pos_des.cpu().numpy()

        # Flatten if 2D (e.g., [1, 7] -> [7])
        arm_joint_pos_des = arm_joint_pos_des.flatten()

        # Validate values
        position_list = []
        for x in arm_joint_pos_des:
            val = float(x)
            if math.isnan(val) or math.isinf(val):
                self.get_logger().warn(f'Invalid arm joint position detected: {val}, replacing with 0.0')
                val = 0.0
            position_list.append(val)

        # Primary: JointState on /teleop_joint_commands (for TeleopJointImpedanceController)
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = self.ARM_JOINT_NAMES[:len(position_list)]
        msg.position = position_list
        msg.velocity = []
        msg.effort = []
        self.arm_joint_pos_pub.publish(msg)

        # Legacy: Float32MultiArray on /arm_joint_positions (for compatibility)
        legacy_msg = Float32MultiArray()
        legacy_msg.layout = MultiArrayLayout()
        dim = MultiArrayDimension()
        dim.label = "fr3_arm_joints"
        dim.size = len(position_list)
        dim.stride = len(position_list)
        legacy_msg.layout.dim = [dim]
        legacy_msg.layout.data_offset = 0
        legacy_msg.data = position_list
        self.arm_joint_pos_legacy_pub.publish(legacy_msg)


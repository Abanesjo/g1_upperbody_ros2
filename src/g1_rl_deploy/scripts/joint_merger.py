#!/usr/bin/env python3
"""Merges upper body (CBF), lower body (policy), and default-hold joint commands
into a single 29-DOF JointState at 500 Hz for the g1_bridge."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Full 29-DOF joint names in motor index order
JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

DEFAULT_POS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,       # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,        # right leg
     0.0, 0.0, 0.0,                          # waist
     0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,  # left arm
     0.35,-0.18, 0.0, 0.87, 0.0, 0.0, 0.0,  # right arm
]

NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


class JointMergerNode(Node):
    def __init__(self):
        super().__init__('joint_merger')

        self.declare_parameter('publish_rate', 500.0)
        rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # Current merged positions — start at defaults
        self.positions = list(DEFAULT_POS)

        self.pub = self.create_publisher(JointState, '/joint_commands', SENSOR_QOS)

        self.create_subscription(
            JointState, '/upper_body_targets',
            self._upper_body_cb, SENSOR_QOS)

        self.create_subscription(
            JointState, '/lower_body_commands',
            self._lower_body_cb, SENSOR_QOS)

        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            f'Joint merger ready: publishing /joint_commands at {rate:.0f} Hz')

    def _upper_body_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in NAME_TO_IDX and i < len(msg.position):
                self.positions[NAME_TO_IDX[name]] = msg.position[i]

    def _lower_body_cb(self, msg):
        for i, name in enumerate(msg.name):
            if name in NAME_TO_IDX and i < len(msg.position):
                self.positions[NAME_TO_IDX[name]] = msg.position[i]

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = list(self.positions)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointMergerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

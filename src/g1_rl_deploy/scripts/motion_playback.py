#!/usr/bin/env python3
"""Plays back a CSV motion file as /joint_commands for the 8 upper body joints."""
import csv
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# CSV column layout: [0:3] base_pos, [3:7] base_quat, [7:36] joint_pos (29 DOF)
# Joint index i -> CSV column 7+i

# Upper body joints: (name, motor_index)
UPPER_BODY_JOINTS = [
    ("waist_roll_joint", 13),
    ("waist_pitch_joint", 14),
    ("left_shoulder_pitch_joint", 15),
    ("left_shoulder_roll_joint", 16),
    ("left_elbow_joint", 18),
    ("right_shoulder_pitch_joint", 22),
    ("right_shoulder_roll_joint", 23),
    ("right_elbow_joint", 25),
]

CSV_FPS = 30.0  # Native frame rate of the CSV motion data


class MotionPlaybackNode(Node):
    def __init__(self):
        super().__init__("motion_playback")

        self.declare_parameter("motion_file",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/g1_rl_deploy/motions/dance1_subject2.csv")
        self.declare_parameter("fps", CSV_FPS)
        self.declare_parameter("loop", True)

        motion_file = self.get_parameter("motion_file").get_parameter_value().string_value
        self.fps = self.get_parameter("fps").get_parameter_value().double_value
        self.loop = self.get_parameter("loop").get_parameter_value().bool_value

        self.get_logger().info(f"Loading motion: {motion_file}")
        self.frames = self._load_csv(motion_file)
        self.get_logger().info(f"Loaded {len(self.frames)} frames at {self.fps} FPS "
                               f"({len(self.frames)/self.fps:.1f}s), loop={self.loop}")

        self.frame_idx = 0
        self.joint_names = [name for name, _ in UPPER_BODY_JOINTS]
        self.col_indices = [7 + idx for _, idx in UPPER_BODY_JOINTS]

        self.pub = self.create_publisher(JointState, "/joint_commands", 1)
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)

    def _load_csv(self, path):
        frames = []
        with open(path) as f:
            for row in csv.reader(f):
                frames.append([float(v) for v in row])
        return frames

    def publish_frame(self):
        if self.frame_idx >= len(self.frames):
            if self.loop:
                self.frame_idx = 0
                self.get_logger().info("Looping motion")
            else:
                self.get_logger().info("Motion complete")
                self.timer.cancel()
                return

        row = self.frames[self.frame_idx]
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [row[c] for c in self.col_indices]
        self.pub.publish(msg)
        self.frame_idx += 1


def main():
    rclpy.init()
    node = MotionPlaybackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gated CSV playback for the 11 upper-body joint targets."""
import csv

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Joy
from std_msgs.msg import String

# CSV column layout: [0:3] base_pos, [3:7] base_quat, [7:36] joint_pos (29 DOF)
# Joint index i -> CSV column 7+i

# Upper body joints: (name, motor_index)
UPPER_BODY_JOINTS = [
    ("waist_yaw_joint", 12),
    ("waist_roll_joint", 13),
    ("waist_pitch_joint", 14),
    ("left_shoulder_pitch_joint", 15),
    ("left_shoulder_roll_joint", 16),
    ("left_shoulder_yaw_joint", 17),
    ("left_elbow_joint", 18),
    ("right_shoulder_pitch_joint", 22),
    ("right_shoulder_roll_joint", 23),
    ("right_shoulder_yaw_joint", 24),
    ("right_elbow_joint", 25),
]

CSV_FPS = 15.0  # Native frame rate of the CSV motion data
STOP_BUTTON = 9
START_BUTTON = 10
CONTROL_STATE = "control"

DEFAULT_UPPER_BODY_TARGET = [
    0.0,
    0.0,
    0.0,
    0.37,
    0.62,
    0.0,
    0.82,
    0.33,
    -0.67,
    0.0,
    1.01,
]


def _button_pressed(buttons, index):
    return index < len(buttons) and buttons[index] != 0


class MotionPlaybackGate:
    """Pure playback state machine shared by the node and unit tests."""

    def __init__(self, loop=True, orchestrator_required=True):
        self.loop = loop
        self.orchestrator_required = orchestrator_required
        self.active = False
        self.frame_idx = 0
        self.orchestrator_state = None
        self._start_pressed = False
        self._stop_pressed = False

    def stop_and_reset(self):
        self.active = False
        self.frame_idx = 0

    def handle_orchestrator_state(self, state):
        """Return True when stopping active playback needs a safe target."""
        previous_state = self.orchestrator_state
        self.orchestrator_state = state
        if not self.orchestrator_required or previous_state is None:
            return False
        if state == previous_state:
            return False

        was_active = self.active
        self.stop_and_reset()
        return was_active

    def handle_joy(self, buttons):
        """Update the gate and return True when a safe target is requested."""
        start_pressed = _button_pressed(buttons, START_BUTTON)
        stop_pressed = _button_pressed(buttons, STOP_BUTTON)
        start_rising = start_pressed and not self._start_pressed
        stop_rising = stop_pressed and not self._stop_pressed
        self._start_pressed = start_pressed
        self._stop_pressed = stop_pressed

        if stop_rising:
            self.stop_and_reset()
            return True

        # A held stop button also takes precedence over a new start edge.
        if stop_pressed:
            return False

        control_allowed = (
            not self.orchestrator_required
            or self.orchestrator_state == CONTROL_STATE
        )
        if start_rising and control_allowed:
            self.active = True
            self.frame_idx = 0
        return False

    def next_frame(self, frame_count):
        """Return the next frame index, or None when playback is inactive."""
        if not self.active or frame_count <= 0:
            return None

        if self.frame_idx >= frame_count:
            if not self.loop:
                self.stop_and_reset()
                return None
            self.frame_idx = 0

        frame_idx = self.frame_idx
        self.frame_idx += 1
        return frame_idx


class MotionPlaybackNode(Node):
    def __init__(self):
        super().__init__("motion_playback")

        self.declare_parameter(
            "motion_file",
            "/workspace/ros2_ws/install/g1_rl_deploy/share/"
            "g1_rl_deploy/motions/dance1_subject2.csv",
        )
        self.declare_parameter("fps", CSV_FPS)
        self.declare_parameter("loop", True)
        self.declare_parameter("orchestrator_required", True)

        motion_file = (
            self.get_parameter("motion_file")
            .get_parameter_value()
            .string_value
        )
        self.fps = self.get_parameter("fps").get_parameter_value().double_value
        self.loop = self.get_parameter("loop").get_parameter_value().bool_value
        orchestrator_required = (
            self.get_parameter("orchestrator_required")
            .get_parameter_value()
            .bool_value
        )
        if self.fps <= 0.0:
            raise ValueError("fps must be greater than zero")

        self.get_logger().info(f"Loading motion: {motion_file}")
        self.frames = self._load_csv(motion_file)
        if not self.frames:
            raise ValueError(f"Motion file contains no frames: {motion_file}")
        self.get_logger().info(
            f"Loaded {len(self.frames)} frames at {self.fps} FPS "
            f"({len(self.frames) / self.fps:.1f}s), loop={self.loop}"
        )

        self.joint_names = [name for name, _ in UPPER_BODY_JOINTS]
        self.col_indices = [7 + idx for _, idx in UPPER_BODY_JOINTS]
        required_columns = max(self.col_indices) + 1
        for row_index, row in enumerate(self.frames):
            if len(row) < required_columns:
                raise ValueError(
                    f"Motion row {row_index} has {len(row)} columns; "
                    f"at least {required_columns} are required"
                )

        self.gate = MotionPlaybackGate(
            loop=self.loop,
            orchestrator_required=orchestrator_required,
        )
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(
            JointState, "/joint_commands", sensor_qos
        )
        self.create_subscription(Joy, "/joy", self._joy_callback, sensor_qos)
        self.create_subscription(
            String,
            "/orchestrator/state",
            self._orchestrator_state_callback,
            state_qos,
        )
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)
        self.get_logger().info(
            "Motion playback is inactive; press Joy button[10] in control "
            "mode to start"
        )

    def _load_csv(self, path):
        frames = []
        with open(path) as motion_file:
            for row in csv.reader(motion_file):
                frames.append([float(value) for value in row])
        return frames

    def _publish_positions(self, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = list(positions)
        self.pub.publish(msg)

    def _joy_callback(self, msg):
        if self.gate.handle_joy(msg.buttons):
            self._publish_positions(DEFAULT_UPPER_BODY_TARGET)
            self.get_logger().info(
                "Motion stopped and reset by Joy button[9]"
            )

    def _orchestrator_state_callback(self, msg):
        if self.gate.handle_orchestrator_state(msg.data):
            self._publish_positions(DEFAULT_UPPER_BODY_TARGET)
            self.get_logger().info(
                "Motion stopped and reset on orchestrator transition"
            )

    def publish_frame(self):
        frame_idx = self.gate.next_frame(len(self.frames))
        if frame_idx is None:
            return

        row = self.frames[frame_idx]
        self._publish_positions([row[column] for column in self.col_indices])


def main():
    rclpy.init()
    node = MotionPlaybackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

import math
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import String


JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint',
    'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint',
    'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

DEFAULT_UPPER_POSITIONS = [
    0.0, 0.0, 0.0,
    0.37, 0.62, 0.0, 0.82, 0.0, 0.0, 0.0,
    0.33, -0.67, 0.0, 1.01, 0.0, 0.0, 0.0,
]


class OrchestratorProbe(Node):

    def __init__(self):
        super().__init__('orchestrator_cbf_toggle_probe')
        sensor_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        state_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', sensor_qos)
        self.joint_command_pub = self.create_publisher(
            JointState, '/joint_commands', sensor_qos)
        self.joy_pub = self.create_publisher(Joy, '/joy', sensor_qos)
        self.safe_command = None
        self.cbf_enabled = None
        self.orchestrator_state = None
        self.create_subscription(
            JointState,
            '/joint_commands_safe',
            self._safe_command_callback,
            sensor_qos,
        )
        self.create_subscription(
            Bool, '/cbf/enabled', self._cbf_enabled_callback, state_qos)
        self.create_subscription(
            String,
            '/orchestrator/state',
            self._orchestrator_state_callback,
            state_qos,
        )

    def _safe_command_callback(self, msg):
        self.safe_command = msg

    def _cbf_enabled_callback(self, msg):
        self.cbf_enabled = msg.data

    def _orchestrator_state_callback(self, msg):
        self.orchestrator_state = msg.data

    def wait_for(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return
        raise AssertionError('timed out waiting for orchestrator output')

    def publish_robot(self, lower_position, upper_position):
        state = JointState()
        state.name = JOINT_NAMES
        state.position = [0.2] * 12 + [1.5] * 17

        command = JointState()
        command.name = JOINT_NAMES
        command.position = (
            [lower_position] * 12 + [upper_position] * 17)
        command.velocity = [0.4] * 29
        command.effort = [0.5] * 29

        for _ in range(5):
            self.joint_state_pub.publish(state)
            self.joint_command_pub.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)

    def publish_joy(self, *pressed_buttons):
        msg = Joy()
        msg.buttons = [0] * 11
        for index in pressed_buttons:
            msg.buttons[index] = 1
        for _ in range(3):
            self.joy_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.03)


def _close(value, expected, tolerance=0.03):
    return math.isclose(value, expected, abs_tol=tolerance)


def test_orchestrator_initial_neutral_disables_cbf():
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    rclpy.init()
    executable = (
        Path(get_package_prefix('g1_bridge'))
        / 'lib' / 'g1_bridge' / 'g1_orchestrator_node'
    )
    process = subprocess.Popen(
        [str(executable)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = OrchestratorProbe()

    try:
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.orchestrator_state == 'neutral')

        late_cbf_state = {'enabled': None}
        late_subscription = probe.create_subscription(
            Bool,
            '/cbf/enabled',
            lambda msg: late_cbf_state.update(enabled=msg.data),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        probe.wait_for(lambda: late_cbf_state['enabled'] is False)
        assert late_subscription is not None
    finally:
        probe.destroy_node()
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        rclpy.shutdown()


def test_orchestrator_cbf_toggle():
    # Isolate this integration test from ROS nodes running on the host.
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    rclpy.init()
    executable = (
        Path(get_package_prefix('g1_bridge'))
        / 'lib' / 'g1_bridge' / 'g1_orchestrator_node'
    )
    process = subprocess.Popen(
        [
            str(executable),
            '--ros-args', '-p', 'initial_mode:=control',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = OrchestratorProbe()

    try:
        # Receive startup state, then create a second subscriber after that
        # publication to verify transient-local late-join behavior.
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.orchestrator_state == 'control')
        late_cbf_state = {'enabled': None}
        late_subscription = probe.create_subscription(
            Bool,
            '/cbf/enabled',
            lambda msg: late_cbf_state.update(enabled=msg.data),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        probe.wait_for(lambda: late_cbf_state['enabled'] is True)
        assert late_subscription is not None

        probe.publish_robot(0.8, 2.0)
        probe.wait_for(
            lambda: probe.safe_command is not None
            and _close(probe.safe_command.position[0], 0.8)
            and _close(probe.safe_command.position[12], 2.0))

        probe.publish_joy(1)
        probe.wait_for(lambda: probe.cbf_enabled is False)
        assert _close(probe.safe_command.position[0], 0.8)
        assert _close(probe.safe_command.velocity[0], 0.4)
        assert _close(probe.safe_command.effort[0], 0.5)
        assert 0.0 < probe.safe_command.position[12] <= 1.5
        assert probe.safe_command.velocity[12] == 0.0
        assert probe.safe_command.effort[12] == 0.0

        # Lower-body targets stay live while upper targets remain overridden.
        probe.publish_robot(0.9, 2.2)
        probe.wait_for(
            lambda: _close(probe.safe_command.position[0], 0.9))
        assert not _close(probe.safe_command.position[12], 2.2)

        # The ramp completes in the existing three-second neutral duration.
        probe.wait_for(
            lambda: probe.safe_command.position[12] == 0.0,
            timeout=3.5,
        )
        assert all(
            _close(actual, expected)
            for actual, expected in zip(
                probe.safe_command.position[12:],
                DEFAULT_UPPER_POSITIONS,
            )
        )
        probe.publish_joy(1)
        assert _close(probe.safe_command.position[12], 0.0)

        # Enabling resumes the latest target immediately.
        probe.publish_joy(1, 2)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and _close(probe.safe_command.position[12], 2.2))

        # When both edges occur together, enable wins.
        probe.publish_joy()
        probe.publish_joy(1, 2)
        assert probe.cbf_enabled is True
        assert _close(probe.safe_command.position[12], 2.2)

        # Entering neutral gates every CBF off even when it was enabled.
        probe.publish_joy()
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.orchestrator_state == 'neutral')

        # Return to control after the neutral ramp, then verify damp reset.
        probe.wait_for(
            lambda: probe.safe_command.position[12] == 0.0,
            timeout=3.5,
        )
        assert all(
            _close(actual, expected)
            for actual, expected in zip(
                probe.safe_command.position[12:],
                DEFAULT_UPPER_POSITIONS,
            )
        )
        probe.publish_joy()
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.orchestrator_state == 'control')

        # Manual disable/enable remains available after control resumes.
        probe.publish_joy()
        probe.publish_joy(1)
        probe.wait_for(lambda: probe.cbf_enabled is False)
        probe.publish_joy()
        probe.publish_joy(2)
        probe.wait_for(lambda: probe.cbf_enabled is True)
        probe.publish_joy()
        probe.publish_joy(4)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.orchestrator_state == 'damp')
    finally:
        probe.destroy_node()
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        rclpy.shutdown()

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
        self.cbf_events = []
        self.workspace_enable_request = None
        self.workspace_request_events = []
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
            Bool,
            '/cbf/workspace_enable_request',
            self._workspace_enable_request_callback,
            state_qos,
        )
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
        self.cbf_events.append(msg.data)

    def _workspace_enable_request_callback(self, msg):
        self.workspace_enable_request = msg.data
        self.workspace_request_events.append(msg.data)

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


def test_orchestrator_initial_neutral_disables_both_cbf_requests():
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
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'neutral')

        late_state = {'upper': None, 'workspace': None}
        late_subscriptions = [
            probe.create_subscription(
                Bool,
                '/cbf/enabled',
                lambda msg: late_state.update(upper=msg.data),
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/workspace_enable_request',
                lambda msg: late_state.update(workspace=msg.data),
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
        ]
        probe.wait_for(
            lambda: late_state['upper'] is False
            and late_state['workspace'] is False)
        assert all(
            subscription is not None for subscription in late_subscriptions)
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
        # Initial control mode still requires explicit CBF activation. Both
        # retained topics must report false to subscribers that join later.
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'control')
        late_state = {'upper': None, 'workspace': None}
        late_subscriptions = [
            probe.create_subscription(
                Bool,
                '/cbf/enabled',
                lambda msg: late_state.update(upper=msg.data),
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/workspace_enable_request',
                lambda msg: late_state.update(workspace=msg.data),
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
        ]
        probe.wait_for(
            lambda: late_state['upper'] is False
            and late_state['workspace'] is False)
        assert all(
            subscription is not None for subscription in late_subscriptions)

        probe.publish_robot(0.8, 2.0)
        probe.wait_for(
            lambda: probe.safe_command is not None
            and _close(probe.safe_command.position[0], 0.8))
        assert not _close(probe.safe_command.position[12], 2.0)

        # Button 2 enables only the upper CBF path and immediately exposes the
        # latest upper target. The workspace request remains disabled.
        probe.publish_joy(2)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.workspace_enable_request is False
            and _close(probe.safe_command.position[12], 2.0))

        # Button 3 requests workspace recenter/enable without changing joint
        # targets. Button 0 disables only that request.
        probe.publish_joy()
        true_events = sum(probe.workspace_request_events)
        probe.publish_joy(3)
        probe.wait_for(
            lambda: probe.workspace_enable_request is True
            and sum(probe.workspace_request_events) == true_events + 1)
        assert probe.cbf_enabled is True
        assert _close(probe.safe_command.position[12], 2.0)

        probe.publish_joy()
        probe.publish_joy(0)
        probe.wait_for(lambda: probe.workspace_enable_request is False)
        assert probe.cbf_enabled is True
        assert _close(probe.safe_command.position[12], 2.0)

        # Workspace enable wins when buttons 0 and 3 rise together.
        probe.publish_joy()
        true_events = sum(probe.workspace_request_events)
        probe.publish_joy(0, 3)
        probe.wait_for(
            lambda: probe.workspace_enable_request is True
            and sum(probe.workspace_request_events) == true_events + 1)

        # Every new button-3 edge is an event, even while the retained request
        # is already true.
        probe.publish_joy()
        true_events = sum(probe.workspace_request_events)
        probe.publish_joy(3)
        probe.wait_for(
            lambda: sum(probe.workspace_request_events) == true_events + 1)
        probe.publish_joy()
        probe.publish_joy(3)
        probe.wait_for(
            lambda: sum(probe.workspace_request_events) == true_events + 2)
        assert probe.workspace_enable_request is True
        assert probe.cbf_enabled is True

        # Button 1 disables only upper CBF authority. Lower-body targets remain
        # live while the upper body ramps to its default pose.
        probe.publish_joy()
        probe.publish_joy(1)
        probe.wait_for(lambda: probe.cbf_enabled is False)
        assert probe.workspace_enable_request is True
        assert _close(probe.safe_command.position[0], 0.8)
        assert _close(probe.safe_command.velocity[0], 0.4)
        assert _close(probe.safe_command.effort[0], 0.5)
        assert 0.0 < probe.safe_command.position[12] <= 1.5
        assert probe.safe_command.velocity[12] == 0.0
        assert probe.safe_command.effort[12] == 0.0

        probe.publish_robot(0.9, 2.2)
        probe.wait_for(lambda: _close(probe.safe_command.position[0], 0.9))
        assert not _close(probe.safe_command.position[12], 2.2)

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

        # Upper enable wins when buttons 1 and 2 rise together and does not
        # affect the workspace request.
        probe.publish_joy()
        probe.publish_joy(1, 2)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.workspace_enable_request is True
            and _close(probe.safe_command.position[12], 2.2))

        # Every actual state transition publishes false for both interfaces.
        upper_false_events = probe.cbf_events.count(False)
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy()
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'neutral'
            and probe.cbf_events.count(False) == upper_false_events + 1
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1)

        # CBF buttons are ignored outside control mode.
        upper_events = len(probe.cbf_events)
        workspace_events = len(probe.workspace_request_events)
        probe.publish_joy()
        probe.publish_joy(2, 3)
        assert probe.cbf_enabled is False
        assert probe.workspace_enable_request is False
        assert len(probe.cbf_events) == upper_events
        assert len(probe.workspace_request_events) == workspace_events
        probe.publish_joy()

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

        # Entering control republishes both false defaults even though they
        # were already false throughout neutral mode.
        upper_false_events = probe.cbf_events.count(False)
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'control'
            and probe.cbf_events.count(False) == upper_false_events + 1
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1)

        # Each interface can be enabled independently after control resumes.
        probe.publish_joy()
        probe.publish_joy(2)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.workspace_enable_request is False)
        probe.publish_joy()
        probe.publish_joy(3)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.workspace_enable_request is True)

        # Damp has priority, publishes both false, and ignores later buttons.
        upper_false_events = probe.cbf_events.count(False)
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy()
        probe.publish_joy(4)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'damp'
            and probe.cbf_events.count(False) == upper_false_events + 1
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1)
        upper_events = len(probe.cbf_events)
        workspace_events = len(probe.workspace_request_events)
        probe.publish_joy(2, 3)
        assert probe.cbf_enabled is False
        assert probe.workspace_enable_request is False
        assert len(probe.cbf_events) == upper_events
        assert len(probe.workspace_request_events) == workspace_events
    finally:
        probe.destroy_node()
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        rclpy.shutdown()

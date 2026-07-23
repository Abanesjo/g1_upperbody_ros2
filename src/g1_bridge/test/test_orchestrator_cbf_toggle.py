import math
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
from g1_cbf_msg.msg import WorkspaceState
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
    'left_knee_joint', 'left_ankle_pitch_joint',
    'left_ankle_roll_joint', 'right_hip_pitch_joint',
    'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint',
    'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint',
    'waist_pitch_joint', 'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint',
    'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint',
    'right_wrist_yaw_joint',
]

DEFAULT_POSITIONS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.37, 0.62, 0.0, 0.82, 0.0, 0.0, 0.0,
    0.33, -0.67, 0.0, 1.01, 0.0, 0.0, 0.0,
]


def _state_qos():
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class OrchestratorProbe(Node):

    def __init__(self):
        super().__init__('orchestrator_cbf_toggle_probe')
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.joint_state_pub = self.create_publisher(
            JointState, '/joint_states', sensor_qos)
        self.joint_command_pub = self.create_publisher(
            JointState, '/joint_commands', sensor_qos)
        self.joy_pub = self.create_publisher(Joy, '/joy', sensor_qos)
        self.workspace_state_pub = self.create_publisher(
            WorkspaceState, '/cbf/workspace_state', _state_qos())

        self.safe_command = None
        self.cbf_enabled = None
        self.cbf_events = []
        self.external_enabled = None
        self.external_events = []
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
            Bool, '/cbf/enabled', self._cbf_enabled_callback, _state_qos())
        self.create_subscription(
            Bool,
            '/cbf/external_enabled',
            self._external_enabled_callback,
            _state_qos(),
        )
        self.create_subscription(
            Bool,
            '/cbf/workspace_enable_request',
            self._workspace_enable_request_callback,
            _state_qos(),
        )
        self.create_subscription(
            String,
            '/orchestrator/state',
            self._orchestrator_state_callback,
            _state_qos(),
        )

    def _safe_command_callback(self, msg):
        self.safe_command = msg

    def _cbf_enabled_callback(self, msg):
        self.cbf_enabled = msg.data
        self.cbf_events.append(msg.data)

    def _external_enabled_callback(self, msg):
        self.external_enabled = msg.data
        self.external_events.append(msg.data)

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

    def publish_workspace_state(self, enabled, capture_pending):
        msg = WorkspaceState()
        msg.enabled = enabled
        msg.capture_pending = capture_pending
        for _ in range(3):
            self.workspace_state_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.03)


def _close(value, expected, tolerance=0.03):
    return math.isclose(value, expected, abs_tol=tolerance)


def _start_orchestrator(initial_mode=None):
    executable = (
        Path(get_package_prefix('g1_bridge'))
        / 'lib' / 'g1_bridge' / 'g1_orchestrator_node'
    )
    command = [str(executable)]
    if initial_mode is not None:
        command.extend([
            '--ros-args', '-p', f'initial_mode:={initial_mode}',
        ])
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_orchestrator(probe, process):
    probe.destroy_node()
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    rclpy.shutdown()


def test_orchestrator_initial_neutral_retained_gates():
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    rclpy.init()
    process = _start_orchestrator()
    probe = OrchestratorProbe()

    try:
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.external_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'neutral')

        late_state = {
            'mode': None,
            'external': None,
            'workspace': None,
        }
        late_subscriptions = [
            probe.create_subscription(
                Bool,
                '/cbf/enabled',
                lambda msg: late_state.update(mode=msg.data),
                _state_qos(),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/external_enabled',
                lambda msg: late_state.update(external=msg.data),
                _state_qos(),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/workspace_enable_request',
                lambda msg: late_state.update(workspace=msg.data),
                _state_qos(),
            ),
        ]
        probe.wait_for(
            lambda: late_state == {
                'mode': False,
                'external': False,
                'workspace': False,
            })
        assert all(
            subscription is not None for subscription in late_subscriptions)
    finally:
        _stop_orchestrator(probe, process)


def test_orchestrator_independent_cbf_gates_and_control_passthrough():
    os.environ['ROS_DOMAIN_ID'] = str(100 + os.getpid() % 100)
    rclpy.init()
    process = _start_orchestrator('control')
    probe = OrchestratorProbe()

    try:
        # Control mode owns the overall upper/self gate. Human and workspace
        # CBFs still require explicit activation after startup.
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.external_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'control')

        late_state = {
            'mode': None,
            'external': None,
            'workspace': None,
        }
        late_subscriptions = [
            probe.create_subscription(
                Bool,
                '/cbf/enabled',
                lambda msg: late_state.update(mode=msg.data),
                _state_qos(),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/external_enabled',
                lambda msg: late_state.update(external=msg.data),
                _state_qos(),
            ),
            probe.create_subscription(
                Bool,
                '/cbf/workspace_enable_request',
                lambda msg: late_state.update(workspace=msg.data),
                _state_qos(),
            ),
        ]
        probe.wait_for(
            lambda: late_state == {
                'mode': True,
                'external': False,
                'workspace': False,
            })
        assert all(
            subscription is not None for subscription in late_subscriptions)

        # All RL targets pass through even while the external human gate is
        # disabled.
        probe.publish_robot(0.8, 2.0)
        probe.wait_for(
            lambda: probe.safe_command is not None
            and _close(probe.safe_command.position[0], 0.8)
            and _close(probe.safe_command.position[12], 2.0))
        assert _close(probe.safe_command.velocity[12], 0.4)
        assert _close(probe.safe_command.effort[12], 0.5)

        # Button 2 toggles only the external human barriers. Repeated Joy
        # samples while held do not retrigger the toggle.
        mode_event_count = len(probe.cbf_events)
        external_true_events = probe.external_events.count(True)
        probe.publish_joy(2)
        probe.wait_for(
            lambda: probe.external_enabled is True
            and probe.external_events.count(True)
            == external_true_events + 1)
        assert probe.cbf_enabled is True
        assert probe.workspace_enable_request is False
        assert len(probe.cbf_events) == mode_event_count
        external_event_count = len(probe.external_events)
        probe.publish_joy(2)
        assert probe.external_enabled is True
        assert len(probe.external_events) == external_event_count

        probe.publish_joy()
        probe.publish_joy(2)
        probe.wait_for(lambda: probe.external_enabled is False)
        probe.publish_joy()
        probe.publish_joy(2)
        probe.wait_for(lambda: probe.external_enabled is True)
        assert probe.external_enabled is True
        assert probe.cbf_enabled is True

        # Button 3 independently toggles the workspace request. Turning it on
        # requests a capture; the next rising edge turns it off.
        probe.publish_joy()
        workspace_true_events = probe.workspace_request_events.count(True)
        probe.publish_joy(3)
        probe.wait_for(
            lambda: probe.workspace_enable_request is True
            and probe.workspace_request_events.count(True)
            == workspace_true_events + 1)
        workspace_event_count = len(probe.workspace_request_events)
        probe.publish_joy(3)
        assert probe.workspace_enable_request is True
        assert len(probe.workspace_request_events) == workspace_event_count

        # Failed capture feedback resets the toggle once capture is no longer
        # pending. The very next button-3 edge therefore retries with a new
        # true request rather than first toggling the stale request off.
        probe.publish_workspace_state(False, False)
        workspace_true_events = probe.workspace_request_events.count(True)
        probe.publish_joy()
        probe.publish_joy(3)
        probe.wait_for(
            lambda: probe.workspace_enable_request is True
            and probe.workspace_request_events.count(True)
            == workspace_true_events + 1)

        probe.publish_joy()
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy(3)
        probe.wait_for(
            lambda: probe.workspace_enable_request is False
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1)
        probe.publish_joy()
        probe.publish_joy(3)
        probe.wait_for(lambda: probe.workspace_enable_request is True)
        assert probe.external_enabled is True
        assert probe.cbf_enabled is True

        # Buttons 0 and 1 are reserved for center seeking and path tracking;
        # the orchestrator must not consume them as CBF controls.
        probe.publish_joy()
        gate_event_counts = (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )
        probe.publish_joy(0, 1)
        assert probe.external_enabled is True
        assert probe.workspace_enable_request is True
        assert gate_event_counts == (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )

        probe.publish_robot(0.9, 2.2)
        probe.wait_for(
            lambda: _close(probe.safe_command.position[0], 0.9)
            and _close(probe.safe_command.position[12], 2.2))
        assert _close(probe.safe_command.velocity[0], 0.4)
        assert _close(probe.safe_command.velocity[12], 0.4)
        assert _close(probe.safe_command.effort[0], 0.5)
        assert _close(probe.safe_command.effort[12], 0.5)

        # Leaving control disables the mode gate and force-resets both
        # independently controlled gates.
        mode_false_events = probe.cbf_events.count(False)
        external_false_events = probe.external_events.count(False)
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy()
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.external_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'neutral'
            and probe.cbf_events.count(False) == mode_false_events + 1
            and probe.external_events.count(False)
            == external_false_events + 1
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1)

        # CBF buttons are ignored outside control.
        event_counts = (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )
        probe.publish_joy()
        probe.publish_joy(1, 2, 0, 3)
        assert probe.cbf_enabled is False
        assert probe.external_enabled is False
        assert probe.workspace_enable_request is False
        assert event_counts == (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )

        # Neutral remains a whole-body ramp to the existing defaults.
        probe.publish_joy()
        probe.wait_for(
            lambda: probe.safe_command is not None
            and all(
                _close(actual, expected)
                for actual, expected in zip(
                    probe.safe_command.position,
                    DEFAULT_POSITIONS,
                )
            ),
            timeout=3.5,
        )
        assert all(value == 0.0 for value in probe.safe_command.velocity)
        assert all(value == 0.0 for value in probe.safe_command.effort)
        # The position tolerance can be reached one timer tick before the
        # orchestrator clears its internal neutral-ramp flag.
        time.sleep(0.1)
        rclpy.spin_once(probe, timeout_sec=0.02)

        # Entering control enables only the mode gate. External/workspace are
        # force-reset, and the latest full RL command resumes immediately.
        mode_true_events = probe.cbf_events.count(True)
        external_false_events = probe.external_events.count(False)
        workspace_false_events = probe.workspace_request_events.count(False)
        probe.publish_joy(5)
        probe.wait_for(
            lambda: probe.cbf_enabled is True
            and probe.external_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'control'
            and probe.cbf_events.count(True) == mode_true_events + 1
            and probe.external_events.count(False)
            == external_false_events + 1
            and probe.workspace_request_events.count(False)
            == workspace_false_events + 1
            and _close(probe.safe_command.position[12], 2.2))

        # Damp has priority, resets all gates, and preserves the existing damp
        # command behavior.
        probe.publish_joy()
        probe.publish_joy(2)
        probe.wait_for(lambda: probe.external_enabled is True)
        probe.publish_joy()
        probe.publish_joy(3)
        probe.wait_for(lambda: probe.workspace_enable_request is True)
        probe.publish_joy()
        probe.publish_joy(4)
        probe.wait_for(
            lambda: probe.cbf_enabled is False
            and probe.external_enabled is False
            and probe.workspace_enable_request is False
            and probe.orchestrator_state == 'damp'
            and probe.safe_command is not None
            and _close(probe.safe_command.position[0], 0.2)
            and _close(probe.safe_command.position[12], 1.5))
        assert all(value == 0.0 for value in probe.safe_command.velocity)
        assert all(value == 0.0 for value in probe.safe_command.effort)

        event_counts = (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )
        probe.publish_joy(2, 3, 5)
        assert event_counts == (
            len(probe.cbf_events),
            len(probe.external_events),
            len(probe.workspace_request_events),
        )
    finally:
        _stop_orchestrator(probe, process)

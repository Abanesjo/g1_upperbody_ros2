import math
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from unitree_hg.msg import LowCmd
from unitree_hg.msg import LowState


NUM_MOTORS = 29
UPPER_BODY_START = 12
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
JOINT_POSITIONS = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.37, 0.62, 0.0, 0.82, 0.0, 0.0, 0.0,
    0.33, -0.67, 0.0, 1.01, 0.0, 0.0, 0.0,
]

SENSOR_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)
STATE_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class BridgeProbe(Node):

    def __init__(self):
        super().__init__('bridge_gravity_modes_probe')
        self.low_state_pub = self.create_publisher(
            LowState, '/lowstate', SENSOR_QOS)
        self.safe_command_pub = self.create_publisher(
            JointState, '/joint_commands_safe', SENSOR_QOS)
        self.state_pub = self.create_publisher(
            String, '/orchestrator/state', STATE_QOS)
        self.low_cmd = None
        self.create_subscription(
            LowCmd, '/lowcmd', self._low_cmd_callback, SENSOR_QOS)

    def _low_cmd_callback(self, msg):
        self.low_cmd = msg

    def wait_for(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if predicate():
                return
        raise AssertionError('timed out waiting for bridge output')

    def publish_low_state(self):
        msg = LowState()
        msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        for index, position in enumerate(JOINT_POSITIONS):
            msg.motor_state[index].q = position
        for _ in range(5):
            self.low_state_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)

    def publish_safe_command(self):
        msg = JointState()
        msg.name = list(JOINT_NAMES)
        msg.position = list(JOINT_POSITIONS)
        msg.velocity = [0.0] * NUM_MOTORS
        msg.effort = [0.0] * NUM_MOTORS
        for _ in range(5):
            self.safe_command_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)

    def publish_state(self, state):
        msg = String()
        msg.data = state
        for _ in range(5):
            self.state_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.02)


def _upper_torques(command):
    return [
        command.motor_cmd[index].tau
        for index in range(UPPER_BODY_START, NUM_MOTORS)
    ]


def _lower_torques(command):
    return [
        command.motor_cmd[index].tau
        for index in range(UPPER_BODY_START)
    ]


def _all_zero(values, tolerance=1e-6):
    return all(math.isclose(value, 0.0, abs_tol=tolerance) for value in values)


def test_gravity_feedforward_modes():
    os.environ['ROS_DOMAIN_ID'] = str(210 + os.getpid() % 20)
    rclpy.init()
    executable = (
        Path(get_package_prefix('g1_bridge'))
        / 'lib' / 'g1_bridge' / 'g1_bridge_node'
    )
    urdf_path = (
        Path(get_package_share_directory('g1_description'))
        / 'urdf' / 'g1_29_inspire.urdf'
    )
    process = subprocess.Popen(
        [
            str(executable),
            '--ros-args',
            '-p', 'simulator:=true',
            '-p', 'wbc:=false',
            '-p', 'gravity:=true',
            '-p', f'urdf_path:={urdf_path}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = BridgeProbe()

    try:
        probe.publish_state('neutral')
        probe.publish_low_state()
        probe.publish_safe_command()
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and math.isclose(probe.low_cmd.motor_cmd[15].kp, 50.0)
            and max(abs(value) for value in _upper_torques(probe.low_cmd))
            > 1e-3
        )
        assert _all_zero(_lower_torques(probe.low_cmd))

        # Gravity falls back to zero if fresh robot state stops arriving.
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and _all_zero(_upper_torques(probe.low_cmd)),
            timeout=1.0,
        )

        probe.low_cmd = None
        probe.publish_state('control')
        probe.publish_low_state()
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and math.isclose(probe.low_cmd.motor_cmd[15].kp, 100.0)
            and max(abs(value) for value in _upper_torques(probe.low_cmd))
            > 1e-3
        )
        assert _all_zero(_lower_torques(probe.low_cmd))

        probe.low_cmd = None
        probe.publish_state('damp')
        probe.publish_low_state()
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and math.isclose(probe.low_cmd.motor_cmd[15].kp, 0.0)
            and _all_zero(_upper_torques(probe.low_cmd))
        )
        assert _all_zero(_lower_torques(probe.low_cmd))
    finally:
        probe.destroy_node()
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        rclpy.shutdown()


def test_wbc_disables_gravity_feedforward():
    os.environ['ROS_DOMAIN_ID'] = str(210 + os.getpid() % 20)
    rclpy.init()
    executable = (
        Path(get_package_prefix('g1_bridge'))
        / 'lib' / 'g1_bridge' / 'g1_bridge_node'
    )
    process = subprocess.Popen(
        [
            str(executable),
            '--ros-args',
            '-p', 'simulator:=true',
            '-p', 'wbc:=true',
            '-p', 'gravity:=true',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    probe = BridgeProbe()

    try:
        probe.publish_state('neutral')
        probe.publish_low_state()
        probe.publish_safe_command()
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and math.isclose(probe.low_cmd.motor_cmd[15].kp, 50.0)
            and _all_zero(_upper_torques(probe.low_cmd))
        )

        probe.low_cmd = None
        probe.publish_state('control')
        probe.publish_low_state()
        probe.wait_for(
            lambda: probe.low_cmd is not None
            and math.isclose(probe.low_cmd.motor_cmd[15].kp, 100.0)
            and _all_zero(_upper_torques(probe.low_cmd))
        )
    finally:
        probe.destroy_node()
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        rclpy.shutdown()

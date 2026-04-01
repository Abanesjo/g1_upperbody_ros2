#!/usr/bin/env python3
"""Direct RL policy node — bypasses CBF and bridge.

Subscribes to /lowstate for IMU + joint feedback, runs the ONNX policy,
and publishes /lowcmd directly with CRC and PD gains.

Includes a FixStand startup phase that linearly interpolates the robot
from its current pose to the training home pose before engaging the policy.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from unitree_hg.msg import LowState, LowCmd

from g1_upperbody.policy import OnnxPolicy
from g1_upperbody.observation import ObservationBuilder
from g1_upperbody.crc import compute_crc
from g1_upperbody.constants import (
    JOINT_NAMES,
    DEFAULT_JOINT_POS,
    ACTION_SCALE,
    NUM_JOINTS,
    UPPER_BODY_COMMAND_JOINTS,
)

# URDF joint name -> motor index (29-DOF G1, matches bridge)
JOINT_MAP = {
    'left_hip_pitch_joint': 0,
    'left_hip_roll_joint': 1,
    'left_hip_yaw_joint': 2,
    'left_knee_joint': 3,
    'left_ankle_pitch_joint': 4,
    'left_ankle_roll_joint': 5,
    'right_hip_pitch_joint': 6,
    'right_hip_roll_joint': 7,
    'right_hip_yaw_joint': 8,
    'right_knee_joint': 9,
    'right_ankle_pitch_joint': 10,
    'right_ankle_roll_joint': 11,
    'waist_yaw_joint': 12,
    'waist_roll_joint': 13,
    'waist_pitch_joint': 14,
    'left_shoulder_pitch_joint': 15,
    'left_shoulder_roll_joint': 16,
    'left_shoulder_yaw_joint': 17,
    'left_elbow_joint': 18,
    'left_wrist_roll_joint': 19,
    'left_wrist_pitch_joint': 20,
    'left_wrist_yaw_joint': 21,
    'right_shoulder_pitch_joint': 22,
    'right_shoulder_roll_joint': 23,
    'right_shoulder_yaw_joint': 24,
    'right_elbow_joint': 25,
    'right_wrist_roll_joint': 26,
    'right_wrist_pitch_joint': 27,
    'right_wrist_yaw_joint': 28,
}

# Per-joint PD gains from training actuator model (ONNX metadata).
JOINT_STIFFNESS = np.array([
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
    40.179, 28.501, 28.501,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
    14.251, 14.251, 14.251, 14.251, 14.251, 16.778, 16.778,
], dtype=np.float32)

JOINT_DAMPING = np.array([
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
    2.558, 1.814, 1.814,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
    0.907, 0.907, 0.907, 0.907, 0.907, 1.068, 1.068,
], dtype=np.float32)

# States
STATE_FIXSTAND = 0
STATE_POLICY = 1


class G1UpperBodyDirectNode(Node):
    def __init__(self):
        super().__init__('g1_upperbody_direct_node')

        # Parameters
        self.declare_parameter('policy_path', '')
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('phase_period', 0.6)
        self.declare_parameter('kp_scale', 1.0)
        self.declare_parameter('kd_scale', 1.0)
        self.declare_parameter('standup_duration', 3.0)

        policy_path = self.get_parameter('policy_path').value
        dt = self.get_parameter('dt').value
        phase_period = self.get_parameter('phase_period').value
        self._kp_scale = self.get_parameter('kp_scale').value
        self._kd_scale = self.get_parameter('kd_scale').value
        self._standup_duration = self.get_parameter('standup_duration').value

        if not policy_path:
            self.get_logger().fatal('policy_path parameter is required')
            raise RuntimeError('policy_path not set')

        # Load ONNX policy
        self.get_logger().info(f'Loading ONNX policy: {policy_path}')
        self.policy = OnnxPolicy(policy_path)
        self.get_logger().info('Policy loaded')

        # Observation builder
        self.obs_builder = ObservationBuilder(phase_period=phase_period)
        self.dt = dt

        # State from /lowstate
        self._imu_ang_vel = None
        self._imu_quat = None
        self._joint_pos = None
        self._joint_vel = None
        self._mode_pr = 0
        self._mode_machine = 0
        self._has_state = False

        # Command inputs
        self._cmd_vel = np.zeros(3, dtype=np.float32)
        self._upper_body_cmd = np.zeros(len(UPPER_BODY_COMMAND_JOINTS),
                                        dtype=np.float32)

        self._ub_name_to_idx = {
            name: i for i, name in enumerate(UPPER_BODY_COMMAND_JOINTS)
        }

        # FixStand state
        self._state = STATE_FIXSTAND
        self._standup_start_pos = None
        self._standup_elapsed = 0.0

        # Subscribers
        self.create_subscription(
            LowState, '/lowstate', self._lowstate_cb, 10)
        self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(
            JointState, '/joint_commands_raw', self._upper_body_cb, 10)

        # Publisher
        self._lowcmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)

        # Timer
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'g1_upperbody_direct_node ready — {1.0/dt:.0f} Hz, '
            f'standup={self._standup_duration}s')

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _lowstate_cb(self, msg: LowState):
        imu = msg.imu_state
        self._imu_quat = np.array(imu.quaternion, dtype=np.float32)
        self._imu_ang_vel = np.array(imu.gyroscope, dtype=np.float32)

        pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        for j, name in enumerate(JOINT_NAMES):
            idx = JOINT_MAP[name]
            pos[j] = msg.motor_state[idx].q
            vel[j] = msg.motor_state[idx].dq
        self._joint_pos = pos
        self._joint_vel = vel

        self._mode_pr = msg.mode_pr
        self._mode_machine = msg.mode_machine
        self._has_state = True

    def _cmd_vel_cb(self, msg: Twist):
        self._cmd_vel = np.array([
            msg.linear.x, msg.linear.y, msg.angular.z,
        ], dtype=np.float32)

    def _upper_body_cb(self, msg: JointState):
        cmd = self._upper_body_cmd.copy()
        for i, name in enumerate(msg.name):
            if name in self._ub_name_to_idx:
                idx = self._ub_name_to_idx[name]
                if i < len(msg.position):
                    cmd[idx] = msg.position[i]
        self._upper_body_cmd = cmd

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #

    def _publish_cmd(self, q_target):
        """Build and publish LowCmd with PD gains and CRC."""
        cmd = LowCmd()
        cmd.mode_pr = self._mode_pr
        cmd.mode_machine = self._mode_machine

        for j, name in enumerate(JOINT_NAMES):
            idx = JOINT_MAP[name]
            cmd.motor_cmd[idx].mode = 1
            cmd.motor_cmd[idx].q = float(q_target[j])
            cmd.motor_cmd[idx].kp = float(JOINT_STIFFNESS[j] * self._kp_scale)
            cmd.motor_cmd[idx].kd = float(JOINT_DAMPING[j] * self._kd_scale)

        compute_crc(cmd)
        self._lowcmd_pub.publish(cmd)

    def _tick(self):
        if not self._has_state:
            return

        if self._state == STATE_FIXSTAND:
            self._tick_fixstand()
        else:
            self._tick_policy()

    def _tick_fixstand(self):
        """Linearly interpolate from current pose to training home pose."""
        if self._standup_start_pos is None:
            self._standup_start_pos = self._joint_pos.copy()
            self._standup_elapsed = 0.0
            self.get_logger().info('FixStand: moving to home pose...')

        self._standup_elapsed += self.dt
        alpha = min(self._standup_elapsed / self._standup_duration, 1.0)

        q_target = (1.0 - alpha) * self._standup_start_pos + alpha * DEFAULT_JOINT_POS
        self._publish_cmd(q_target)

        if alpha >= 1.0:
            self._state = STATE_POLICY
            self.obs_builder.reset()
            self.get_logger().info('FixStand complete — engaging RL policy')

    def _tick_policy(self):
        """Run RL policy inference and publish commands."""
        obs = self.obs_builder.build(
            imu_ang_vel=self._imu_ang_vel,
            imu_quat=self._imu_quat,
            cmd_vel=self._cmd_vel,
            joint_pos=self._joint_pos,
            joint_vel=self._joint_vel,
            upper_body_cmd=self._upper_body_cmd,
            dt=self.dt,
        )

        actions = self.policy.infer(obs)
        actions = np.clip(actions, -5.0, 5.0)
        self.obs_builder.update_last_actions(actions)

        q_target = DEFAULT_JOINT_POS + actions * ACTION_SCALE
        self._publish_cmd(q_target)


def main(args=None):
    rclpy.init(args=args)
    node = G1UpperBodyDirectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

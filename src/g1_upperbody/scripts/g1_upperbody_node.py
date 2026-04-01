#!/usr/bin/env python3
"""RL policy node for G1 upper body + velocity control.

Runs a trained ONNX policy at 50 Hz. Subscribes to IMU, joint states,
velocity commands, and upper-body position commands; publishes 29-DOF
joint position targets on /joint_commands_unsafe.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from unitree_hg.msg import LowState

from g1_upperbody.policy import OnnxPolicy
from g1_upperbody.observation import ObservationBuilder
from g1_upperbody.constants import (
    JOINT_NAMES,
    DEFAULT_JOINT_POS,
    ACTION_SCALE,
    NUM_JOINTS,
    UPPER_BODY_COMMAND_JOINTS,
)


class G1UpperBodyNode(Node):
    def __init__(self):
        super().__init__('g1_upperbody_node')

        # Parameters
        self.declare_parameter('policy_path', '')
        self.declare_parameter('dt', 0.02)
        self.declare_parameter('phase_period', 0.6)

        policy_path = self.get_parameter('policy_path').value
        dt = self.get_parameter('dt').value
        phase_period = self.get_parameter('phase_period').value

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

        # State storage (populated by callbacks)
        self._imu_ang_vel = None
        self._imu_quat = None
        self._joint_pos = None
        self._joint_vel = None
        self._cmd_vel = np.zeros(3, dtype=np.float32)
        self._upper_body_cmd = np.zeros(len(UPPER_BODY_COMMAND_JOINTS),
                                        dtype=np.float32)

        # Name-to-index maps for parsing incoming JointState messages
        self._joint_name_to_idx = {
            name: i for i, name in enumerate(JOINT_NAMES)
        }
        self._ub_name_to_idx = {
            name: i for i, name in enumerate(UPPER_BODY_COMMAND_JOINTS)
        }

        # Subscribers
        self.create_subscription(
            LowState, '/lowstate', self._lowstate_cb, 10)
        self.create_subscription(
            JointState, '/joint_states', self._joint_states_cb, 10)
        self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self.create_subscription(
            JointState, '/joint_commands_raw', self._upper_body_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(
            JointState, '/joint_commands_unsafe', 10)

        # Timer
        self.create_timer(dt, self._tick)

        self.get_logger().info(
            f'g1_upperbody_node ready - publishing at {1.0/dt:.0f} Hz')

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _lowstate_cb(self, msg: LowState):
        imu = msg.imu_state
        # Unitree/MuJoCo quaternion order: (w, x, y, z)
        self._imu_quat = np.array(imu.quaternion, dtype=np.float32)
        self._imu_ang_vel = np.array(imu.gyroscope, dtype=np.float32)

    def _joint_states_cb(self, msg: JointState):
        pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(msg.name):
            if name in self._joint_name_to_idx:
                idx = self._joint_name_to_idx[name]
                if i < len(msg.position):
                    pos[idx] = msg.position[i]
                if i < len(msg.velocity):
                    vel[idx] = msg.velocity[i]
        self._joint_pos = pos
        self._joint_vel = vel

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

    def _tick(self):
        # Wait until we have IMU and joint state data
        if self._imu_ang_vel is None or self._joint_pos is None:
            return

        # Build observation vector
        obs = self.obs_builder.build(
            imu_ang_vel=self._imu_ang_vel,
            imu_quat=self._imu_quat,
            cmd_vel=self._cmd_vel,
            joint_pos=self._joint_pos,
            joint_vel=self._joint_vel,
            upper_body_cmd=self._upper_body_cmd,
            dt=self.dt,
        )

        # Run ONNX inference
        actions = self.policy.infer(obs)

        # Store actions for next observation
        self.obs_builder.update_last_actions(actions)

        # Convert to joint position targets: q = default + action * scale
        q_target = DEFAULT_JOINT_POS + actions * ACTION_SCALE

        # Publish all 29 joints
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = list(JOINT_NAMES)
        cmd_msg.position = q_target.tolist()
        self.cmd_pub.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = G1UpperBodyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
G1 Bridge Node: bridges Unitree G1 lowstate/lowcmd to standard ROS2 interfaces.

Subscriptions:
  /lowstate      (unitree_hg/msg/LowState) -> publishes /joint_states, /imu
  /joint_commands (sensor_msgs/JointState)  -> publishes /lowcmd

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Imu
from unitree_hg.msg import LowState, LowCmd

from g1_bridge.crc import compute_crc

G1_NUM_MOTOR = 29

# Best-effort, volatile, depth 1 — matches Unitree SDK DDS defaults
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# URDF joint name -> motor index (29DOF version)
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

# Ordered joint names (sorted by motor index) for consistent publishing
JOINT_NAMES = sorted(JOINT_MAP.keys(), key=lambda n: JOINT_MAP[n])


class G1BridgeNode(Node):
    def __init__(self):
        super().__init__('g1_bridge_node')

        # Load gains from parameters
        self._gains = {}
        for name in JOINT_NAMES:
            self.declare_parameter(f'gains.{name}.kp', 100.0)
            self.declare_parameter(f'gains.{name}.kd', 3.0)
            kp = self.get_parameter(f'gains.{name}.kp').value
            kd = self.get_parameter(f'gains.{name}.kd').value
            self._gains[name] = (kp, kd)

        # State tracking
        self._mode_machine = 0
        self._mode_pr = 0
        self._has_state = False

        self._latest_lowcmd = None

        # Publishers
        self._joint_states_pub = self.create_publisher(JointState, '/joint_states', SENSOR_QOS)
        self._imu_pub = self.create_publisher(Imu, '/imu', SENSOR_QOS)
        self._lowcmd_pub = self.create_publisher(LowCmd, '/lowcmd', SENSOR_QOS)

        # Subscribers
        self._lowstate_sub = self.create_subscription(
            LowState, '/lowstate', self._lowstate_cb, SENSOR_QOS)
        self._joint_cmd_sub = self.create_subscription(
            JointState, '/joint_commands', self._joint_cmd_cb, SENSOR_QOS)

        # 500 Hz republish timer for stable PD control
        self._republish_timer = self.create_timer(0.002, self._republish_lowcmd)

        self.get_logger().info('G1 bridge node started')

    def _lowstate_cb(self, msg: LowState):
        self._mode_machine = msg.mode_machine
        self._mode_pr = msg.mode_pr
        self._has_state = True

        stamp = self.get_clock().now().to_msg()

        # Build and publish JointState
        js = JointState()
        js.header.stamp = stamp

        for name in JOINT_NAMES:
            idx = JOINT_MAP[name]
            motor = msg.motor_state[idx]
            js.name.append(name)
            js.position.append(float(motor.q))
            js.velocity.append(float(motor.dq))
            js.effort.append(float(motor.tau_est))

        self._joint_states_pub.publish(js)

        # Build and publish Imu
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = 'torso_link'
        # Quaternion: unitree convention [w,x,y,z] -> ROS Imu [x,y,z,w]
        imu.orientation.w = float(msg.imu_state.quaternion[0])
        imu.orientation.x = float(msg.imu_state.quaternion[1])
        imu.orientation.y = float(msg.imu_state.quaternion[2])
        imu.orientation.z = float(msg.imu_state.quaternion[3])
        imu.angular_velocity.x = float(msg.imu_state.gyroscope[0])
        imu.angular_velocity.y = float(msg.imu_state.gyroscope[1])
        imu.angular_velocity.z = float(msg.imu_state.gyroscope[2])
        imu.linear_acceleration.x = float(msg.imu_state.accelerometer[0])
        imu.linear_acceleration.y = float(msg.imu_state.accelerometer[1])
        imu.linear_acceleration.z = float(msg.imu_state.accelerometer[2])

        self._imu_pub.publish(imu)

    def _joint_cmd_cb(self, msg: JointState):
        if not self._has_state:
            self.get_logger().warn(
                'Received /joint_commands before any /lowstate — ignoring')
            return

        cmd = LowCmd()
        cmd.mode_pr = self._mode_pr
        cmd.mode_machine = self._mode_machine

        # Enable all 29 motors with configured gains
        for name in JOINT_NAMES:
            idx = JOINT_MAP[name]
            kp, kd = self._gains[name]
            cmd.motor_cmd[idx].mode = 1
            cmd.motor_cmd[idx].kp = float(kp)
            cmd.motor_cmd[idx].kd = float(kd)

        # Apply commanded values from JointState message
        has_pos = len(msg.position) > 0
        has_vel = len(msg.velocity) > 0
        has_eff = len(msg.effort) > 0

        for i, name in enumerate(msg.name):
            if name not in JOINT_MAP:
                self.get_logger().warn(f'Unknown joint name in command: {name}')
                continue
            idx = JOINT_MAP[name]
            if has_pos and i < len(msg.position):
                cmd.motor_cmd[idx].q = float(msg.position[i])
            if has_vel and i < len(msg.velocity):
                cmd.motor_cmd[idx].dq = float(msg.velocity[i])
            if has_eff and i < len(msg.effort):
                cmd.motor_cmd[idx].tau = float(msg.effort[i])

        compute_crc(cmd)
        self._latest_lowcmd = cmd
        self._lowcmd_pub.publish(cmd)

    def _republish_lowcmd(self):
        if self._latest_lowcmd is not None:
            self._lowcmd_pub.publish(self._latest_lowcmd)


def main(args=None):
    rclpy.init(args=args)
    node = G1BridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Publish base-frame cmd_vel commands that drive pelvis pose to world origin."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class OriginCmdVelPublisherNode(Node):
    def __init__(self):
        super().__init__('origin_cmd_vel_publisher_node')

        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('target_yaw', 0.0)
        self.declare_parameter('kp_linear', 0.5)
        self.declare_parameter('kd_linear', 0.05)
        self.declare_parameter('kp_yaw', 1.0)
        self.declare_parameter('kd_yaw', 0.05)
        self.declare_parameter('position_deadzone', 0.1)
        self.declare_parameter('yaw_deadzone_deg', 15.0)
        self.declare_parameter('max_linear_x', 1.0)
        self.declare_parameter('max_reverse_x', 0.5)
        self.declare_parameter('max_linear_y', 0.5)
        self.declare_parameter('max_angular_z', 1.0)
        self.declare_parameter('pose_timeout_sec', 0.2)

        self._rate_hz = float(self.get_parameter('rate_hz').value)
        self._target_xy = np.array([
            float(self.get_parameter('target_x').value),
            float(self.get_parameter('target_y').value),
        ], dtype=np.float64)
        self._target_yaw = float(self.get_parameter('target_yaw').value)
        self._kp_linear = float(self.get_parameter('kp_linear').value)
        self._kd_linear = float(self.get_parameter('kd_linear').value)
        self._kp_yaw = float(self.get_parameter('kp_yaw').value)
        self._kd_yaw = float(self.get_parameter('kd_yaw').value)
        self._position_deadzone = float(
            self.get_parameter('position_deadzone').value
        )
        yaw_deadzone_deg = float(
            self.get_parameter('yaw_deadzone_deg').value
        )
        self._yaw_deadzone = math.radians(yaw_deadzone_deg)
        self._max_linear_x = float(self.get_parameter('max_linear_x').value)
        self._max_reverse_x = float(self.get_parameter('max_reverse_x').value)
        self._max_linear_y = float(self.get_parameter('max_linear_y').value)
        self._max_angular_z = float(
            self.get_parameter('max_angular_z').value
        )
        self._pose_timeout_sec = float(
            self.get_parameter('pose_timeout_sec').value
        )

        self._validate_params()

        self._xy = None
        self._yaw = None
        self._last_pose_time = None
        self._last_xy = None
        self._last_yaw = None
        self._last_velocity_time = None
        self._xy_vel_world = np.zeros(2, dtype=np.float64)
        self._yaw_rate = 0.0

        self.create_subscription(
            PoseStamped, '/pose/pelvis', self._pose_cb, SENSOR_QOS,
        )
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', SENSOR_QOS)
        self.create_timer(1.0 / self._rate_hz, self._tick)

        self.get_logger().info(
            'origin_cmd_vel_publisher_node ready: '
            f'target_xy={self._target_xy.tolist()}, '
            f'target_yaw={self._target_yaw:.3f}, '
            f'position_deadzone={self._position_deadzone:.3f}, '
            f'yaw_deadzone={yaw_deadzone_deg:.1f}deg'
        )

    def _validate_params(self):
        if self._rate_hz <= 0.0:
            raise ValueError('rate_hz must be positive')
        if self._kp_linear < 0.0:
            raise ValueError('kp_linear must be non-negative')
        if self._kd_linear < 0.0:
            raise ValueError('kd_linear must be non-negative')
        if self._kp_yaw < 0.0:
            raise ValueError('kp_yaw must be non-negative')
        if self._kd_yaw < 0.0:
            raise ValueError('kd_yaw must be non-negative')
        if self._position_deadzone < 0.0:
            raise ValueError('position_deadzone must be non-negative')
        if self._yaw_deadzone < 0.0:
            raise ValueError('yaw_deadzone_deg must be non-negative')
        if self._max_linear_x < 0.0:
            raise ValueError('max_linear_x must be non-negative')
        if self._max_reverse_x < 0.0:
            raise ValueError('max_reverse_x must be non-negative')
        if self._max_linear_y < 0.0:
            raise ValueError('max_linear_y must be non-negative')
        if self._max_angular_z < 0.0:
            raise ValueError('max_angular_z must be non-negative')
        if self._pose_timeout_sec < 0.0:
            raise ValueError('pose_timeout_sec must be non-negative')

    def _pose_cb(self, msg: PoseStamped):
        now = self.get_clock().now()
        xy = np.array(
            [msg.pose.position.x, msg.pose.position.y],
            dtype=np.float64,
        )
        yaw = self._yaw_from_quat(msg.pose.orientation)

        if self._last_velocity_time is not None:
            dt = (now - self._last_velocity_time).nanoseconds * 1e-9
            if dt > 1e-6:
                self._xy_vel_world = (xy - self._last_xy) / dt
                self._yaw_rate = self._wrap_angle(yaw - self._last_yaw) / dt

        self._xy = xy
        self._yaw = yaw
        self._last_pose_time = now
        self._last_xy = xy
        self._last_yaw = yaw
        self._last_velocity_time = now

    def _tick(self):
        cmd = Twist()

        if not self._pose_ready():
            self.get_logger().warn(
                'origin cmd_vel missing or stale /pose/pelvis; '
                'publishing zero /cmd_vel',
                throttle_duration_sec=2.0,
            )
            self._cmd_pub.publish(cmd)
            return

        position_error = self._target_xy - self._xy
        distance = float(np.linalg.norm(position_error))
        if distance > self._position_deadzone:
            v_world = (
                self._kp_linear * position_error
                - self._kd_linear * self._xy_vel_world
            )
            v_base = self._world_xy_to_base(v_world, self._yaw)
            cmd.linear.x = float(
                np.clip(v_base[0], -self._max_reverse_x, self._max_linear_x)
            )
            cmd.linear.y = float(
                np.clip(v_base[1], -self._max_linear_y, self._max_linear_y)
            )

        yaw_error = self._wrap_angle(self._target_yaw - self._yaw)
        if abs(yaw_error) > self._yaw_deadzone:
            yaw_cmd = self._kp_yaw * yaw_error - self._kd_yaw * self._yaw_rate
            cmd.angular.z = float(
                np.clip(yaw_cmd, -self._max_angular_z, self._max_angular_z)
            )

        self._cmd_pub.publish(cmd)

    def _pose_ready(self):
        if self._xy is None or self._yaw is None:
            return False
        if self._pose_timeout_sec == 0.0:
            return True

        age = (
            self.get_clock().now() - self._last_pose_time
        ).nanoseconds * 1e-9
        return age <= self._pose_timeout_sec

    @staticmethod
    def _yaw_from_quat(quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _world_xy_to_base(v_world, yaw):
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return np.array([
            cos_yaw * v_world[0] + sin_yaw * v_world[1],
            -sin_yaw * v_world[0] + cos_yaw * v_world[1],
        ], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = OriginCmdVelPublisherNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

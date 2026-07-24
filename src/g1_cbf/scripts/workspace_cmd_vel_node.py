#!/usr/bin/env python3
"""Arbitrate joystick, workspace-path, and center-seeking commands."""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Joy
from std_msgs.msg import String

from g1_cbf.tf_pose import (
    TfPoseLookup,
    normalize_frame,
    resolve_lookup_timeout_sec,
)
from g1_cbf_msg.msg import WorkspaceState


WORKSPACE_FRAME = 'workspace'
PELVIS_FRAME = 'pelvis'
BUTTON_CENTER = 0
BUTTON_PATH = 1

AUTHORITY_JOYSTICK = 'joystick'
AUTHORITY_PATH = 'path'
AUTHORITY_CENTER = 'center'

ROUTE_POINTS = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 2.5),
    (-1.0, 2.5),
    (-1.0, -2.5),
    (1.0, -2.5),
    (1.0, 0.0),
    (0.0, 0.0),
)

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

REFERENCE_POINT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(quat):
    x, y, z, w = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def workspace_to_body(vector, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array([
        cos_yaw * vector[0] + sin_yaw * vector[1],
        -sin_yaw * vector[0] + cos_yaw * vector[1],
    ], dtype=np.float64)


def interpolate_path(linear_resolution=0.05):
    samples = [ROUTE_POINTS[0]]
    for previous, current in zip(ROUTE_POINTS[:-1], ROUTE_POINTS[1:]):
        start_xy = np.array(previous, dtype=np.float64)
        end_xy = np.array(current, dtype=np.float64)
        distance = float(np.linalg.norm(end_xy - start_xy))
        count = max(1, int(math.ceil(distance / linear_resolution)))
        for index in range(1, count + 1):
            ratio = index / count
            xy = start_xy + ratio * (end_xy - start_xy)
            samples.append((float(xy[0]), float(xy[1])))
    return samples


class RectangleFollower:
    """Follow a target that moves along the route at a fixed rate."""

    def __init__(self, speed=0.25, tracking_radius=0.1,
                 heading_kp=1.5, max_yaw_rate=1.0):
        self.speed = speed
        self.tracking_radius = tracking_radius
        self.heading_kp = heading_kp
        self.max_yaw_rate = max_yaw_rate
        self._points = np.asarray(ROUTE_POINTS, dtype=np.float64)
        self._segments = self._points[1:] - self._points[:-1]
        self._segment_lengths = np.linalg.norm(self._segments, axis=1)
        self._cumulative_lengths = np.concatenate((
            np.zeros(1, dtype=np.float64),
            np.cumsum(self._segment_lengths),
        ))
        self.total_length = float(self._cumulative_lengths[-1])
        self.reset()

    def reset(self):
        self.target_distance = 0.0
        self.finished = False

    @property
    def target(self):
        target, _ = self._target_and_tangent(self.target_distance)
        return target

    def command(self, position, yaw, elapsed_sec=0.0):
        position = np.asarray(position, dtype=np.float64)
        zero = np.zeros(2, dtype=np.float64)
        if self.finished:
            return zero, 0.0

        if not math.isfinite(elapsed_sec) or elapsed_sec < 0.0:
            raise ValueError('elapsed_sec must be finite and non-negative')
        self.target_distance = min(
            self.total_length,
            self.target_distance + self.speed * elapsed_sec,
        )
        target, tangent = self._target_and_tangent(self.target_distance)
        delta = target - position
        distance = float(np.linalg.norm(delta))

        at_final_target = self.target_distance >= self.total_length
        if at_final_target and distance <= self.tracking_radius:
            self.finished = True
            return zero, 0.0

        follow_tangent = (
            not at_final_target and distance <= self.tracking_radius
        )
        direction = tangent if follow_tangent else delta / distance
        velocity = self.speed * direction
        desired_yaw = math.atan2(direction[1], direction[0])
        yaw_error = wrap_angle(desired_yaw - yaw)
        yaw_rate = float(np.clip(
            self.heading_kp * yaw_error,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))
        return velocity, yaw_rate

    def _target_and_tangent(self, distance):
        if distance >= self.total_length:
            index = len(self._segments) - 1
            return self._points[-1].copy(), (
                self._segments[index] / self._segment_lengths[index]
            )
        index = int(np.searchsorted(
            self._cumulative_lengths[1:],
            distance,
            side='right',
        ))
        along = distance - self._cumulative_lengths[index]
        tangent = self._segments[index] / self._segment_lengths[index]
        return self._points[index] + along * tangent, tangent


class WorkspaceCmdVelNode(Node):
    def __init__(self):
        super().__init__('workspace_cmd_vel_node')

        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('workspace_frame', WORKSPACE_FRAME)
        self.declare_parameter('pelvis_frame', PELVIS_FRAME)
        self.declare_parameter('path_speed', 0.25)
        self.declare_parameter('tracking_radius', 0.1)
        self.declare_parameter('path_resolution', 0.05)
        self.declare_parameter('path_heading_kp', 1.5)
        self.declare_parameter('center_kp_linear', 0.5)
        self.declare_parameter('center_kd_linear', 0.05)
        self.declare_parameter('center_kp_yaw', 1.0)
        self.declare_parameter('center_kd_yaw', 0.05)
        self.declare_parameter('center_position_deadzone', 0.05)
        self.declare_parameter('center_yaw_deadzone_deg', 15.0)
        self.declare_parameter('max_linear_x', 0.5)
        self.declare_parameter('max_linear_y', 0.5)
        self.declare_parameter('max_angular_z', 1.0)
        self.declare_parameter('workspace_cbf_available', True)
        self.declare_parameter('joy_cmd_timeout_sec', 0.5)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_stale_timeout_sec', 0.5)
        self.declare_parameter('tf_timeout_sec', 0.0)
        self.declare_parameter('orchestrator_required', True)

        self._read_params()
        self._validate_params()

        self._pose_lookup = TfPoseLookup(
            self,
            self._workspace_frame,
            self._pelvis_frame,
            resolve_lookup_timeout_sec(self),
        )
        self._follower = RectangleFollower(
            speed=self._path_speed,
            tracking_radius=self._tracking_radius,
            heading_kp=self._path_heading_kp,
            max_yaw_rate=self._max_angular_z,
        )

        self._authority = AUTHORITY_JOYSTICK
        self._orchestrator_state = (
            None if self._orchestrator_required else 'control'
        )
        self._buttons_armed = not self._orchestrator_required
        self._last_center_button = False
        self._last_path_button = False
        self._latest_joy_cmd = Twist()
        self._last_joy_cmd_time = None
        self._workspace_enabled = False
        self._capture_pending = False
        self._workspace_generation = None
        self._capture_pause = False
        self._await_tf_stamp_ns = None
        self._last_path_tick_time = None
        self._last_center_xy = None
        self._last_center_yaw = None
        self._last_center_time = None

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', SENSOR_QOS)
        self._path_pub = self.create_publisher(
            Path, '/workspace_path', STATE_QOS
        )
        self._planar_reference_pub = self.create_publisher(
            PointStamped,
            '/cbf/planar_reference_point',
            REFERENCE_POINT_QOS,
        )
        self.create_subscription(
            Twist, '/cmd_vel_joy', self._joy_cmd_cb, SENSOR_QOS
        )
        self.create_subscription(Joy, '/joy', self._joy_cb, SENSOR_QOS)
        self.create_subscription(
            String,
            '/orchestrator/state',
            self._orchestrator_state_cb,
            STATE_QOS,
        )
        self.create_subscription(
            WorkspaceState,
            '/cbf/workspace_state',
            self._workspace_state_cb,
            STATE_QOS,
        )
        self.create_timer(1.0 / self._rate_hz, self._tick)

        self._publish_path()
        self.get_logger().info(
            'Workspace velocity authority ready: joystick default, '
            'buttons[1]=path toggle, buttons[0]=center toggle'
        )

    def _read_params(self):
        def value(name):
            return self.get_parameter(name).value
        self._rate_hz = float(value('rate_hz'))
        self._workspace_frame = normalize_frame(
            value('workspace_frame'), WORKSPACE_FRAME
        )
        self._pelvis_frame = normalize_frame(
            value('pelvis_frame'), PELVIS_FRAME
        )
        self._path_speed = float(value('path_speed'))
        self._tracking_radius = float(value('tracking_radius'))
        self._path_resolution = float(value('path_resolution'))
        self._path_heading_kp = float(value('path_heading_kp'))
        self._center_kp_linear = float(value('center_kp_linear'))
        self._center_kd_linear = float(value('center_kd_linear'))
        self._center_kp_yaw = float(value('center_kp_yaw'))
        self._center_kd_yaw = float(value('center_kd_yaw'))
        self._center_position_deadzone = float(
            value('center_position_deadzone')
        )
        self._center_yaw_deadzone = math.radians(
            float(value('center_yaw_deadzone_deg'))
        )
        self._max_linear_x = float(value('max_linear_x'))
        self._max_linear_y = float(value('max_linear_y'))
        self._max_angular_z = float(value('max_angular_z'))
        self._workspace_cbf_available = bool(
            value('workspace_cbf_available')
        )
        self._joy_cmd_timeout_sec = float(value('joy_cmd_timeout_sec'))
        self._tf_stale_timeout_sec = float(value('tf_stale_timeout_sec'))
        self._orchestrator_required = bool(value('orchestrator_required'))

    def _validate_params(self):
        positive = {
            'rate_hz': self._rate_hz,
            'path_speed': self._path_speed,
            'tracking_radius': self._tracking_radius,
            'path_resolution': self._path_resolution,
            'path_heading_kp': self._path_heading_kp,
            'max_linear_x': self._max_linear_x,
            'max_linear_y': self._max_linear_y,
            'max_angular_z': self._max_angular_z,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')
        nonnegative = {
            'joy_cmd_timeout_sec': self._joy_cmd_timeout_sec,
            'tf_stale_timeout_sec': self._tf_stale_timeout_sec,
        }
        for name, value in nonnegative.items():
            if value < 0.0:
                raise ValueError(f'{name} must be non-negative')

    def _joy_cmd_cb(self, msg):
        self._latest_joy_cmd = msg
        self._last_joy_cmd_time = self.get_clock().now()

    def _joy_cb(self, msg):
        center_pressed = self._button_pressed(msg, BUTTON_CENTER)
        path_pressed = self._button_pressed(msg, BUTTON_PATH)
        center_event = center_pressed and not self._last_center_button
        path_event = path_pressed and not self._last_path_button
        self._last_center_button = center_pressed
        self._last_path_button = path_pressed

        if not self._control_available():
            return
        if not self._buttons_armed:
            if not center_pressed and not path_pressed:
                self._buttons_armed = True
            return

        center_eligible = (
            self._workspace_cbf_available
            and self._workspace_enabled
            and not self._capture_pending
        )
        if center_event:
            if self._authority == AUTHORITY_CENTER:
                self._set_authority(AUTHORITY_JOYSTICK)
                return
            if center_eligible:
                self._follower.reset()
                self._set_authority(AUTHORITY_CENTER)
                return

        if self._authority == AUTHORITY_CENTER:
            return
        if path_event:
            if self._authority == AUTHORITY_PATH:
                self._set_authority(AUTHORITY_JOYSTICK)
            else:
                self._set_authority(AUTHORITY_PATH)

    def _orchestrator_state_cb(self, msg):
        new_state = str(msg.data)
        if new_state == self._orchestrator_state:
            return
        self._orchestrator_state = new_state
        self._set_authority(AUTHORITY_JOYSTICK)
        self._buttons_armed = False

    def _workspace_state_cb(self, msg):
        generation = int(msg.generation)
        generation_changed = (
            self._workspace_generation is not None
            and generation != self._workspace_generation
        )
        self._workspace_generation = generation
        self._workspace_enabled = bool(msg.enabled)
        self._capture_pending = bool(msg.capture_pending)

        if generation_changed:
            self._follower.reset()
            self._pause_path_progress()
            self._reset_center_history()
            self._await_tf_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
            self._publish_path()

        self._capture_pause = self._capture_pending
        if self._capture_pause:
            self._pause_path_progress()
        if (
            not self._capture_pending
            and not self._workspace_enabled
            and self._authority == AUTHORITY_CENTER
        ):
            self._set_authority(AUTHORITY_JOYSTICK)

    def _set_authority(self, authority):
        if authority == self._authority:
            return
        self._authority = authority
        self._pause_path_progress()
        self._reset_center_history()
        self.get_logger().info(f'cmd_vel authority: {authority}')

    def _control_available(self):
        return (
            not self._orchestrator_required
            or self._orchestrator_state == 'control'
        )

    def _tick(self):
        cmd = Twist()
        if not self._control_available():
            self._pause_path_progress()
            self._cmd_pub.publish(cmd)
            return

        if self._authority == AUTHORITY_JOYSTICK:
            self._pause_path_progress()
            cmd = self._fresh_joy_command()
            self._cmd_pub.publish(cmd)
            return

        if self._capture_pause:
            self._pause_path_progress()
            self._cmd_pub.publish(cmd)
            return

        pose, reason = self._lookup_pose()
        if pose is None:
            self.get_logger().warn(
                f'Autonomous cmd_vel paused: {reason}',
                throttle_duration_sec=2.0,
            )
            self._pause_path_progress()
            self._cmd_pub.publish(cmd)
            return

        position = pose.position[:2]
        yaw = yaw_from_quat(pose.quat)
        if self._authority == AUTHORITY_PATH:
            velocity_ws, yaw_rate = self._follower.command(
                position,
                yaw,
                self._path_elapsed_sec(),
            )
            self._publish_planar_reference(self._follower.target)
            velocity_body = workspace_to_body(velocity_ws, yaw)
            cmd.linear.x = float(np.clip(
                velocity_body[0], -self._max_linear_x, self._max_linear_x
            ))
            cmd.linear.y = float(np.clip(
                velocity_body[1], -self._max_linear_y, self._max_linear_y
            ))
            cmd.angular.z = float(np.clip(
                yaw_rate, -self._max_angular_z, self._max_angular_z
            ))
        else:
            self._pause_path_progress()
            cmd = self._center_command(position, yaw)
        self._cmd_pub.publish(cmd)

    def _fresh_joy_command(self):
        if self._last_joy_cmd_time is None:
            return Twist()
        age = (
            self.get_clock().now() - self._last_joy_cmd_time
        ).nanoseconds * 1e-9
        if self._joy_cmd_timeout_sec > 0.0 and age > self._joy_cmd_timeout_sec:
            return Twist()
        return self._copy_twist(self._latest_joy_cmd)

    def _lookup_pose(self):
        pose, reason = self._pose_lookup.lookup()
        if pose is None:
            return None, (
                f'TF {self._pose_lookup.describe()} unavailable: {reason}'
            )
        values = np.concatenate([pose.position, pose.quat])
        if not np.all(np.isfinite(values)):
            return None, f'TF {self._pose_lookup.describe()} is non-finite'
        if self._tf_stale_timeout_sec > 0.0:
            age = self._pose_lookup.age_sec(pose)
            if age is not None and age > self._tf_stale_timeout_sec:
                return None, (
                    f'TF {self._pose_lookup.describe()} stale by {age:.3f}s'
                )
        if self._await_tf_stamp_ns is not None:
            if (
                pose.stamp.nanoseconds != 0
                and pose.stamp.nanoseconds < self._await_tf_stamp_ns
            ):
                return None, 'waiting for TF from the latest workspace capture'
            self._await_tf_stamp_ns = None
        return pose, ''

    def _center_command(self, position, yaw):
        now = self.get_clock().now()
        velocity_ws = np.zeros(2, dtype=np.float64)
        yaw_rate_measured = 0.0
        if self._last_center_time is not None:
            dt = (now - self._last_center_time).nanoseconds * 1e-9
            if dt > 1e-6:
                velocity_ws = (
                    position - self._last_center_xy
                ) / dt
                yaw_rate_measured = wrap_angle(
                    yaw - self._last_center_yaw
                ) / dt

        position_error = -position
        desired_ws = np.zeros(2, dtype=np.float64)
        if np.linalg.norm(position_error) > self._center_position_deadzone:
            desired_ws = (
                self._center_kp_linear * position_error
                - self._center_kd_linear * velocity_ws
            )
        desired_body = workspace_to_body(desired_ws, yaw)

        yaw_error = wrap_angle(-yaw)
        desired_yaw_rate = 0.0
        if abs(yaw_error) > self._center_yaw_deadzone:
            desired_yaw_rate = (
                self._center_kp_yaw * yaw_error
                - self._center_kd_yaw * yaw_rate_measured
            )

        self._last_center_xy = position.copy()
        self._last_center_yaw = yaw
        self._last_center_time = now

        cmd = Twist()
        cmd.linear.x = float(np.clip(
            desired_body[0], -self._max_linear_x, self._max_linear_x
        ))
        cmd.linear.y = float(np.clip(
            desired_body[1], -self._max_linear_y, self._max_linear_y
        ))
        cmd.angular.z = float(np.clip(
            desired_yaw_rate, -self._max_angular_z, self._max_angular_z
        ))
        return cmd

    def _reset_center_history(self):
        self._last_center_xy = None
        self._last_center_yaw = None
        self._last_center_time = None

    def _pause_path_progress(self):
        self._last_path_tick_time = None

    def _path_elapsed_sec(self):
        now = self.get_clock().now()
        elapsed_sec = 0.0
        if self._last_path_tick_time is not None:
            elapsed_sec = max(
                0.0,
                (now - self._last_path_tick_time).nanoseconds * 1e-9,
            )
        self._last_path_tick_time = now
        return elapsed_sec

    def _publish_planar_reference(self, target):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._workspace_frame
        msg.point.x = float(target[0])
        msg.point.y = float(target[1])
        msg.point.z = 0.0
        self._planar_reference_pub.publish(msg)

    def _publish_path(self):
        stamp = self.get_clock().now().to_msg()
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self._workspace_frame
        samples = interpolate_path(self._path_resolution)
        for x, y in samples:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._workspace_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._path_pub.publish(msg)

    @staticmethod
    def _copy_twist(source):
        cmd = Twist()
        cmd.linear.x = source.linear.x
        cmd.linear.y = source.linear.y
        cmd.linear.z = source.linear.z
        cmd.angular.x = source.angular.x
        cmd.angular.y = source.angular.y
        cmd.angular.z = source.angular.z
        return cmd

    @staticmethod
    def _button_pressed(msg, index):
        return index < len(msg.buttons) and msg.buttons[index] != 0


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceCmdVelNode()
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

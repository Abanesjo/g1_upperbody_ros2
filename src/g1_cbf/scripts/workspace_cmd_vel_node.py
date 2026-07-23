#!/usr/bin/env python3
"""Arbitrate joystick, workspace-path, and center-seeking commands."""

from dataclasses import dataclass
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
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

KEY_POSES = (
    (0.0, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (0.8, 0.0, 90.0),
    (0.8, 2.0, 90.0),
    (0.8, 2.0, 180.0),
    (-0.8, 2.0, 180.0),
    (-0.8, 2.0, 270.0),
    (-0.8, -2.0, 270.0),
    (-0.8, -2.0, 0.0),
    (0.8, -2.0, 0.0),
    (0.8, -2.0, 90.0),
    (0.8, 0.0, 90.0),
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


def _clip_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm < 1e-12:
        return vector
    return vector * (limit / norm)


def _point_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - start))
    ratio = float(np.dot(point - start, segment) / length_sq)
    projection = start + np.clip(ratio, 0.0, 1.0) * segment
    return float(np.linalg.norm(point - projection))


@dataclass(frozen=True)
class MotionPhase:
    kind: str
    start: np.ndarray
    end: np.ndarray
    yaw: float


def build_motion_phases():
    phases = []
    for previous, current in zip(KEY_POSES[:-1], KEY_POSES[1:]):
        start = np.array(previous[:2], dtype=np.float64)
        end = np.array(current[:2], dtype=np.float64)
        yaw = math.radians(current[2])
        kind = 'translate' if np.linalg.norm(end - start) > 1e-9 else 'rotate'
        phases.append(MotionPhase(kind, start, end, yaw))
    return tuple(phases)


def interpolate_path(linear_resolution=0.05, yaw_resolution_deg=5.0):
    samples = [KEY_POSES[0]]
    yaw_resolution = math.radians(yaw_resolution_deg)
    for previous, current in zip(KEY_POSES[:-1], KEY_POSES[1:]):
        start_xy = np.array(previous[:2], dtype=np.float64)
        end_xy = np.array(current[:2], dtype=np.float64)
        distance = float(np.linalg.norm(end_xy - start_xy))
        start_yaw = math.radians(previous[2])
        yaw_delta = wrap_angle(math.radians(current[2]) - start_yaw)
        if distance > 1e-9:
            count = max(1, int(math.ceil(distance / linear_resolution)))
        else:
            count = max(1, int(math.ceil(abs(yaw_delta) / yaw_resolution)))
        for index in range(1, count + 1):
            ratio = index / count
            xy = start_xy + ratio * (end_xy - start_xy)
            yaw = wrap_angle(start_yaw + ratio * yaw_delta)
            samples.append((float(xy[0]), float(xy[1]), yaw))
    return samples


class RectangleFollower:
    """Ordered straight/turn controller that tolerates CBF corner detours."""

    def __init__(self, speed=0.5, tracking_radius=0.1,
                 cross_track_gain=1.0, max_cross_track_speed=0.2,
                 yaw_kp=1.5, yaw_tolerance=math.radians(5.0),
                 max_yaw_rate=1.0):
        self.speed = speed
        self.tracking_radius = tracking_radius
        self.cross_track_gain = cross_track_gain
        self.max_cross_track_speed = max_cross_track_speed
        self.yaw_kp = yaw_kp
        self.yaw_tolerance = yaw_tolerance
        self.max_yaw_rate = max_yaw_rate
        self.phases = build_motion_phases()
        self.reset()

    def reset(self):
        self.initial_pose_complete = False
        self.phase_index = 0
        self.phase_progress = 0.0
        self.finished = False

    def command(self, position, yaw, cbf_active=False, safe_radius=None):
        position = np.asarray(position, dtype=np.float64)
        zero = np.zeros(2, dtype=np.float64)
        if self.finished:
            return zero, 0.0

        if not self.initial_pose_complete:
            target = np.array(KEY_POSES[0][:2], dtype=np.float64)
            delta = target - position
            yaw_error = wrap_angle(math.radians(KEY_POSES[0][2]) - yaw)
            if (
                np.linalg.norm(delta) <= self.tracking_radius
                and abs(yaw_error) <= self.yaw_tolerance
            ):
                self.initial_pose_complete = True
                return zero, 0.0
            velocity = _clip_norm(delta, self.speed)
            yaw_rate = float(np.clip(
                self.yaw_kp * yaw_error,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            ))
            return velocity, yaw_rate

        if self.phase_index >= len(self.phases):
            self.finished = True
            return zero, 0.0

        phase = self.phases[self.phase_index]
        if phase.kind == 'rotate':
            yaw_error = wrap_angle(phase.yaw - yaw)
            if abs(yaw_error) <= self.yaw_tolerance:
                self.phase_index += 1
                self.phase_progress = 0.0
                if self.phase_index >= len(self.phases):
                    self.finished = True
                return zero, 0.0
            return zero, float(np.clip(
                self.yaw_kp * yaw_error,
                -self.max_yaw_rate,
                self.max_yaw_rate,
            ))

        segment = phase.end - phase.start
        length = float(np.linalg.norm(segment))
        tangent = segment / length
        along = float(np.dot(position - phase.start, tangent))
        self.phase_progress = max(self.phase_progress, along)
        endpoint_reached = (
            np.linalg.norm(position - phase.end) <= self.tracking_radius
        )
        endpoint_passed = along >= length
        next_segment_reached = self._next_segment_is_closer(
            position, phase, length
        )
        if endpoint_reached or endpoint_passed or next_segment_reached:
            self.phase_index += 1
            self.phase_progress = 0.0
            if self.phase_index >= len(self.phases):
                self.finished = True
            return zero, 0.0

        projection = phase.start + along * tangent
        cross_track_error = position - projection
        cbf_detour_expected = (
            cbf_active
            and safe_radius is not None
            and safe_radius > 0.0
            and np.linalg.norm(phase.end) > safe_radius
        )
        if cbf_detour_expected:
            # Let the CBF project the pure segment direction onto its circle.
            # Pulling laterally toward an unreachable rectangle edge can
            # otherwise cancel that tangential motion and stall the robot.
            correction = zero
        else:
            correction = _clip_norm(
                -self.cross_track_gain * cross_track_error,
                self.max_cross_track_speed,
            )
        velocity = self.speed * tangent + correction
        yaw_error = wrap_angle(phase.yaw - yaw)
        yaw_rate = float(np.clip(
            self.yaw_kp * yaw_error,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))
        return velocity, yaw_rate

    def _next_segment_is_closer(self, position, phase, length):
        # The rectangle deliberately extends beyond the circular CBF.  At an
        # unreachable corner the filtered motion follows the circle instead
        # of ever reaching the phase endpoint.  Once the robot is past the
        # segment midpoint and geometrically closer to the following segment,
        # hand off the nominal direction so the CBF can keep carrying it
        # around the perimeter.
        handoff_progress = max(self.tracking_radius, 0.5 * length)
        if self.phase_progress < handoff_progress:
            return False
        next_phase = None
        for candidate in self.phases[self.phase_index + 1:]:
            if candidate.kind == 'translate':
                next_phase = candidate
                break
        if next_phase is None:
            return False
        current_distance = _point_segment_distance(
            position, phase.start, phase.end
        )
        next_distance = _point_segment_distance(
            position, next_phase.start, next_phase.end
        )
        return next_distance < current_distance


class WorkspaceCmdVelNode(Node):
    def __init__(self):
        super().__init__('workspace_cmd_vel_node')

        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('workspace_frame', WORKSPACE_FRAME)
        self.declare_parameter('pelvis_frame', PELVIS_FRAME)
        self.declare_parameter('path_speed', 0.5)
        self.declare_parameter('tracking_radius', 0.1)
        self.declare_parameter('path_resolution', 0.05)
        self.declare_parameter('path_yaw_resolution_deg', 5.0)
        self.declare_parameter('cross_track_gain', 1.0)
        self.declare_parameter('max_cross_track_speed', 0.2)
        self.declare_parameter('path_yaw_kp', 1.5)
        self.declare_parameter('path_yaw_tolerance_deg', 5.0)
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
        self.declare_parameter('workspace_safe_radius', 1.4)
        self.declare_parameter('joy_cmd_timeout_sec', 0.5)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_stale_timeout_sec', 0.2)
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
            cross_track_gain=self._cross_track_gain,
            max_cross_track_speed=self._max_cross_track_speed,
            yaw_kp=self._path_yaw_kp,
            yaw_tolerance=self._path_yaw_tolerance,
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
        self._last_center_xy = None
        self._last_center_yaw = None
        self._last_center_time = None

        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', SENSOR_QOS)
        self._path_pub = self.create_publisher(
            Path, '/workspace_path', STATE_QOS
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
        self._path_yaw_resolution_deg = float(
            value('path_yaw_resolution_deg')
        )
        self._cross_track_gain = float(value('cross_track_gain'))
        self._max_cross_track_speed = float(
            value('max_cross_track_speed')
        )
        self._path_yaw_kp = float(value('path_yaw_kp'))
        self._path_yaw_tolerance = math.radians(
            float(value('path_yaw_tolerance_deg'))
        )
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
        self._workspace_safe_radius = float(value('workspace_safe_radius'))
        self._joy_cmd_timeout_sec = float(value('joy_cmd_timeout_sec'))
        self._tf_stale_timeout_sec = float(value('tf_stale_timeout_sec'))
        self._orchestrator_required = bool(value('orchestrator_required'))

    def _validate_params(self):
        positive = {
            'rate_hz': self._rate_hz,
            'path_speed': self._path_speed,
            'tracking_radius': self._tracking_radius,
            'path_resolution': self._path_resolution,
            'path_yaw_resolution_deg': self._path_yaw_resolution_deg,
            'max_linear_x': self._max_linear_x,
            'max_linear_y': self._max_linear_y,
            'max_angular_z': self._max_angular_z,
            'workspace_safe_radius': self._workspace_safe_radius,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')
        nonnegative = {
            'cross_track_gain': self._cross_track_gain,
            'max_cross_track_speed': self._max_cross_track_speed,
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
            self._reset_center_history()
            self._await_tf_stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
            self._publish_path()

        self._capture_pause = self._capture_pending
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
            self._cmd_pub.publish(cmd)
            return

        if self._authority == AUTHORITY_JOYSTICK:
            cmd = self._fresh_joy_command()
            self._cmd_pub.publish(cmd)
            return

        if self._capture_pause:
            self._cmd_pub.publish(cmd)
            return

        pose, reason = self._lookup_pose()
        if pose is None:
            self.get_logger().warn(
                f'Autonomous cmd_vel paused: {reason}',
                throttle_duration_sec=2.0,
            )
            self._cmd_pub.publish(cmd)
            return

        position = pose.position[:2]
        yaw = yaw_from_quat(pose.quat)
        if self._authority == AUTHORITY_PATH:
            velocity_ws, yaw_rate = self._follower.command(
                position,
                yaw,
                cbf_active=(
                    self._workspace_cbf_available
                    and self._workspace_enabled
                ),
                safe_radius=self._workspace_safe_radius,
            )
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

    def _publish_path(self):
        stamp = self.get_clock().now().to_msg()
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = self._workspace_frame
        samples = interpolate_path(
            self._path_resolution,
            self._path_yaw_resolution_deg,
        )
        for x, y, yaw in samples:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._workspace_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
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

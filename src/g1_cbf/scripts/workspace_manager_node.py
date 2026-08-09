#!/usr/bin/env python3
"""Capture and publish the planar CBF workspace frame."""

from dataclasses import dataclass
import math
import time

import rclpy
from geometry_msgs.msg import Transform, TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from g1_cbf.tf_pose import TfPoseLookup, normalize_frame
from g1_cbf_msg.msg import WorkspaceState


STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass
class WorkspaceCaptureState:
    """Deterministic state machine for one-shot workspace capture."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    enabled: bool = False
    generation: int = 0
    pending: bool = False
    deadline: float = 0.0

    def request_enable(self, now, retry_timeout_sec):
        """Disable the old state while a fresh pelvis pose is captured."""
        self.enabled = False
        self.pending = True
        self.deadline = float(now) + float(retry_timeout_sec)

    def request_disable(self):
        """Cancel capture and retain the last transform in disabled state."""
        self.enabled = False
        self.pending = False

    def expire(self, now):
        """Disable a pending request once its retry window has elapsed."""
        if not self.pending or float(now) < self.deadline:
            return False
        self.enabled = False
        self.pending = False
        return True

    def try_capture(self, pose, age_sec, stale_timeout_sec):
        """Commit a valid, fresh pelvis planar pose as a new generation."""
        if not self.pending:
            return False, 'no workspace capture is pending'

        values = [*pose.position, *pose.quat]
        if not all(math.isfinite(float(value)) for value in values):
            return False, 'world-to-pelvis TF contains non-finite values'

        if age_sec is not None:
            age_sec = float(age_sec)
            if not math.isfinite(age_sec) or age_sec < 0.0:
                return False, 'world-to-pelvis TF has an invalid timestamp'
            if stale_timeout_sec > 0.0 and age_sec > stale_timeout_sec:
                return (
                    False,
                    f'world-to-pelvis TF is stale by {age_sec:.3f}s',
                )

        try:
            yaw = yaw_from_quat(pose.quat)
        except ValueError as exc:
            return False, str(exc)

        self.x = float(pose.position[0])
        self.y = float(pose.position[1])
        self.yaw = yaw
        self.generation += 1
        self.enabled = True
        self.pending = False
        return True, ''


def yaw_from_quat(quat):
    """Return yaw from an arbitrary finite, non-zero XYZW quaternion."""

    x, y, z, w = (float(value) for value in quat)
    norm_sq = x * x + y * y + z * z + w * w
    if norm_sq < 1.0e-12:
        raise ValueError('world-to-pelvis TF has a zero-norm quaternion')

    sin_yaw = 2.0 * (w * z + x * y) / norm_sq
    cos_yaw = 1.0 - 2.0 * (y * y + z * z) / norm_sq
    return math.atan2(sin_yaw, cos_yaw)


def workspace_transform(state):
    """Build the planar transform represented by ``state``."""
    transform = Transform()
    transform.translation.x = state.x
    transform.translation.y = state.y
    transform.translation.z = 0.0
    transform.rotation.x = 0.0
    transform.rotation.y = 0.0
    transform.rotation.z = math.sin(0.5 * state.yaw)
    transform.rotation.w = math.cos(0.5 * state.yaw)
    return transform


class WorkspaceManagerNode(Node):
    def __init__(self):
        super().__init__('workspace_manager_node')

        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('workspace_frame', 'workspace')
        self.declare_parameter('pelvis_frame', 'pelvis')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('capture_retry_timeout_sec', 0.5)
        self.declare_parameter('tf_lookup_timeout_sec', 0.0)
        self.declare_parameter('tf_stale_timeout_sec', 0.5)

        self._world_frame = normalize_frame(
            self.get_parameter('world_frame').value,
            'world',
        )
        self._workspace_frame = normalize_frame(
            self.get_parameter('workspace_frame').value,
            'workspace',
        )
        self._pelvis_frame = normalize_frame(
            self.get_parameter('pelvis_frame').value,
            'pelvis',
        )
        self._publish_rate = float(
            self.get_parameter('publish_rate').value
        )
        self._capture_retry_timeout_sec = float(
            self.get_parameter('capture_retry_timeout_sec').value
        )
        self._tf_lookup_timeout_sec = float(
            self.get_parameter('tf_lookup_timeout_sec').value
        )
        self._tf_stale_timeout_sec = float(
            self.get_parameter('tf_stale_timeout_sec').value
        )
        self._validate_params()

        self._capture = WorkspaceCaptureState()
        self._last_capture_error = 'world-to-pelvis TF is unavailable'
        self._pelvis_lookup = TfPoseLookup(
            self,
            self._world_frame,
            self._pelvis_frame,
            self._tf_lookup_timeout_sec,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._state_pub = self.create_publisher(
            WorkspaceState,
            '/cbf/workspace_state',
            STATE_QOS,
        )
        self.create_subscription(
            Bool,
            '/cbf/workspace_enable_request',
            self._enable_request_cb,
            STATE_QOS,
        )
        self.create_timer(1.0 / self._publish_rate, self._tick)

        self._publish_state()
        self._broadcast_transform()
        self.get_logger().info(
            'Workspace manager ready: '
            f'{self._world_frame} -> {self._workspace_frame}, '
            f'pelvis_frame={self._pelvis_frame}, '
            f'rate={self._publish_rate:.1f} Hz'
        )

    def _validate_params(self):
        frames = (
            self._world_frame,
            self._workspace_frame,
            self._pelvis_frame,
        )
        if any(not frame for frame in frames):
            raise ValueError('workspace frame names must not be empty')
        if self._world_frame == self._workspace_frame:
            raise ValueError('world_frame and workspace_frame must differ')
        if self._publish_rate <= 0.0:
            raise ValueError('publish_rate must be positive')
        if self._capture_retry_timeout_sec < 0.0:
            raise ValueError(
                'capture_retry_timeout_sec must be non-negative'
            )
        if self._tf_lookup_timeout_sec < 0.0:
            raise ValueError('tf_lookup_timeout_sec must be non-negative')
        if self._tf_stale_timeout_sec < 0.0:
            raise ValueError('tf_stale_timeout_sec must be non-negative')

    def _enable_request_cb(self, msg):
        if not msg.data:
            self._capture.request_disable()
            self._publish_state()
            self.get_logger().info('Workspace disabled')
            return

        self._capture.request_enable(
            time.monotonic(),
            self._capture_retry_timeout_sec,
        )
        self._last_capture_error = 'world-to-pelvis TF is unavailable'
        self._publish_state()
        self._attempt_capture(ignore_deadline=True)

    def _tick(self):
        if self._capture.pending:
            self._attempt_capture()
        self._broadcast_transform()

    def _attempt_capture(self, ignore_deadline=False):
        now = time.monotonic()
        if not ignore_deadline and self._capture.expire(now):
            self._publish_state()
            self.get_logger().warn(
                'Workspace capture timed out; workspace remains disabled. '
                f'Last error: {self._last_capture_error}. '
                'Publish a new true enable request to retry.'
            )
            return

        pose, reason = self._pelvis_lookup.lookup()
        if pose is None:
            self._last_capture_error = reason or (
                'world-to-pelvis TF is unavailable'
            )
            self.get_logger().warn(
                f'Workspace capture waiting for TF '
                f'{self._pelvis_lookup.describe()}: '
                f'{self._last_capture_error}',
                throttle_duration_sec=2.0,
            )
            return

        age_sec = self._pelvis_lookup.age_sec(pose)
        captured, reason = self._capture.try_capture(
            pose,
            age_sec,
            self._tf_stale_timeout_sec,
        )
        if not captured:
            self._last_capture_error = reason
            self.get_logger().warn(
                f'Workspace capture rejected: {reason}',
                throttle_duration_sec=2.0,
            )
            return

        self._publish_state()
        self._broadcast_transform()
        self.get_logger().info(
            f'Workspace generation {self._capture.generation} captured at '
            f'world XY=({self._capture.x:.3f}, {self._capture.y:.3f}), '
            f'yaw={self._capture.yaw:.3f} rad'
        )

    def _publish_state(self):
        msg = WorkspaceState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.child_frame_id = self._workspace_frame
        msg.transform = workspace_transform(self._capture)
        msg.enabled = self._capture.enabled
        msg.capture_pending = self._capture.pending
        msg.generation = self._capture.generation
        self._state_pub.publish(msg)

    def _broadcast_transform(self):
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._world_frame
        msg.child_frame_id = self._workspace_frame
        msg.transform = workspace_transform(self._capture)
        self._tf_broadcaster.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

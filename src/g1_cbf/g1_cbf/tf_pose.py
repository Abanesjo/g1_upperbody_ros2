"""Helpers for reading robot pose from TF."""

from dataclasses import dataclass

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass
class FramePose:
    """Pose of source_frame expressed in target_frame."""

    position: np.ndarray
    quat: np.ndarray
    stamp: Time


def normalize_frame(frame_id, default=''):
    frame = str(frame_id or default).strip()
    return frame[1:] if frame.startswith('/') else frame


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def resolve_lookup_timeout_sec(node):
    lookup_timeout = float(
        node.get_parameter('tf_lookup_timeout_sec').value
    )
    legacy_timeout = float(node.get_parameter('tf_timeout_sec').value)
    if lookup_timeout > 0.0 or legacy_timeout <= 0.0:
        return lookup_timeout
    return legacy_timeout


class TfPoseLookup:
    """Lookup the latest transform between two configured frames."""

    def __init__(self, node, target_frame, source_frame, timeout_sec):
        self._node = node
        self.target_frame = normalize_frame(target_frame)
        self.source_frame = normalize_frame(source_frame)
        self._timeout = Duration(seconds=max(0.0, float(timeout_sec)))
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)

    @property
    def buffer(self):
        """Underlying TF buffer for other frame-aware inputs on this node."""
        return self._buffer

    def lookup(self):
        try:
            transform = self._buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                Time(),
                timeout=self._timeout,
            )
        except TransformException as exc:
            return None, str(exc)

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        pose = FramePose(
            position=np.array([
                translation.x,
                translation.y,
                translation.z,
            ], dtype=np.float64),
            quat=normalize_quat(np.array([
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ], dtype=np.float64)),
            stamp=Time.from_msg(transform.header.stamp),
        )
        return pose, ''

    def age_sec(self, pose):
        if pose.stamp.nanoseconds == 0:
            return None
        age = (
            self._node.get_clock().now() - pose.stamp
        ).nanoseconds * 1e-9
        if age < 0.0:
            # A future-dated dynamic transform is not fresh evidence. Return
            # infinity so every enabled stale-timeout gate fails closed.
            return float('inf')
        return age

    def describe(self):
        return f'{self.target_frame} -> {self.source_frame}'

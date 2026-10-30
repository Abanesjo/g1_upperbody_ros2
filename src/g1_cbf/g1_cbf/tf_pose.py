"""Helpers for reading robot pose from TF."""

from dataclasses import dataclass

import numpy as np
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
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


# How far a transform stamp may lead the node clock and still count as fresh.
# Under simulation time the pose stamps run up to a few milliseconds ahead of
# the /clock a node has processed.
DEFAULT_FUTURE_TOLERANCE_SEC = 0.02


class TfPoseLookup:
    """Lookup the latest transform between two configured frames."""

    def __init__(self, node, target_frame, source_frame, timeout_sec,
                 future_tolerance_sec=DEFAULT_FUTURE_TOLERANCE_SEC):
        self._node = node
        self.target_frame = normalize_frame(target_frame)
        self.source_frame = normalize_frame(source_frame)
        self._timeout = Duration(seconds=max(0.0, float(timeout_sec)))
        self._future_tolerance_sec = max(0.0, float(future_tolerance_sec))
        self._buffer = Buffer()
        # On a thread of its own, with a private node (None): on the owning
        # node's executor, a control tick lets only a few /tf messages through
        # per cycle, so the queue filled and every lookup returned a pose as
        # old as the queue is long. That is 0.5 s at depth 100, which trips
        # the stale gates. The small depth bounds the age if this thread ever
        # falls behind. Do not pass the owning node with spin_thread: it would
        # then be spun by two executors, control ticks included.
        self._listener = TransformListener(
            self._buffer,
            None,
            spin_thread=True,
            qos=QoSProfile(
                depth=10,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )

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
        if age < -self._future_tolerance_sec:
            # A future-dated dynamic transform is not fresh evidence. Return
            # infinity so every enabled stale-timeout gate fails closed.
            return float('inf')
        return max(age, 0.0)

    def describe(self):
        return f'{self.target_frame} -> {self.source_frame}'

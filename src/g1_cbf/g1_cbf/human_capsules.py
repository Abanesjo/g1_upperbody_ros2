"""Validation and TF conversion for human capsule messages."""

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time

from g1_cbf_msg.msg import Capsule, CapsuleArray


def transform_capsule_array(msg, target_frame, tf_buffer):
    """Return a validated copy of ``msg`` expressed in ``target_frame``."""
    source_frame = str(msg.header.frame_id).strip()
    target_frame = str(target_frame).strip()
    if not source_frame:
        raise ValueError('CapsuleArray.header.frame_id must not be empty')
    if not target_frame:
        raise ValueError('target frame must not be empty')

    capsules = []
    for source in msg.capsules:
        endpoints = np.array([
            [source.a.x, source.a.y, source.a.z],
            [source.b.x, source.b.y, source.b.z],
        ], dtype=np.float64)
        radius = float(source.radius)
        if not np.all(np.isfinite(endpoints)):
            raise ValueError('capsule endpoints must be finite')
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError('capsule radius must be finite and positive')
        capsules.append((source, endpoints, radius))

    transform = None
    if capsules and source_frame != target_frame:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(msg.header.stamp),
            timeout=Duration(),
        )

    result = CapsuleArray()
    result.header.stamp = msg.header.stamp
    result.header.frame_id = target_frame

    for source, endpoints, radius in capsules:
        if transform is not None:
            endpoints = _transform_points(endpoints, transform)

        capsule = Capsule()
        capsule.a.x, capsule.a.y, capsule.a.z = endpoints[0]
        capsule.b.x, capsule.b.y, capsule.b.z = endpoints[1]
        capsule.radius = radius
        capsule.name = source.name
        result.capsules.append(capsule)

    return result


def human_data_is_fresh(last_update, now, timeout_sec):
    """Return whether a successfully received human sample is still usable."""
    if last_update is None:
        return False
    if timeout_sec <= 0.0:
        return True
    age = (now - last_update).nanoseconds * 1e-9
    return 0.0 <= age <= timeout_sec


def human_data_time(header_stamp, receipt_time):
    """Use measurement time when provided, otherwise local receipt time."""
    measurement_time = Time.from_msg(header_stamp)
    if measurement_time.nanoseconds == 0:
        return receipt_time
    return measurement_time


def _transform_points(points, transform):
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    offset = np.array([
        translation.x,
        translation.y,
        translation.z,
    ], dtype=np.float64)
    quat = np.array([
        rotation.x,
        rotation.y,
        rotation.z,
        rotation.w,
    ], dtype=np.float64)
    if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(quat)):
        raise ValueError('TF transform must be finite')
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        raise ValueError('TF rotation quaternion must be nonzero')
    quat /= norm

    xyz = quat[:3]
    twice_cross = 2.0 * np.cross(xyz, points)
    rotated = points + quat[3] * twice_cross + np.cross(xyz, twice_cross)
    transformed = rotated + offset
    if not np.all(np.isfinite(transformed)):
        raise ValueError('transformed capsule endpoints must be finite')
    return transformed

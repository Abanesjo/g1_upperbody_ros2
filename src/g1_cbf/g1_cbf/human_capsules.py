"""Validation and TF conversion for human capsule messages."""

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time

from g1_cbf_msg.msg import Capsule, CapsuleArray


def capsule_arrays(msg, target_frame, tf_buffer):
    """Return the validated capsules of ``msg`` in ``target_frame`` as arrays.

    Returns endpoints (n, 2, 3) (a, b) and radii (n,). Each capsule is read
    once and all of them are validated and transformed together, which keeps
    a message with a crowd of humans cheap to parse.
    """
    source_frame = str(msg.header.frame_id).strip()
    target_frame = str(target_frame).strip()
    if not source_frame:
        raise ValueError('CapsuleArray.header.frame_id must not be empty')
    if not target_frame:
        raise ValueError('target frame must not be empty')

    values = np.array(
        [
            (c.a.x, c.a.y, c.a.z, c.b.x, c.b.y, c.b.z, c.radius)
            for c in msg.capsules
        ],
        dtype=np.float64,
    ).reshape(-1, 7)
    endpoints = values[:, :6].reshape(-1, 2, 3)
    radii = values[:, 6]
    if not np.all(np.isfinite(endpoints)):
        raise ValueError('capsule endpoints must be finite')
    if not np.all(np.isfinite(radii) & (radii > 0.0)):
        raise ValueError('capsule radius must be finite and positive')

    if len(radii) and source_frame != target_frame:
        transform = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time.from_msg(msg.header.stamp),
            timeout=Duration(),
        )
        endpoints = _transform_points(
            endpoints.reshape(-1, 3), transform,
        ).reshape(-1, 2, 3)
    return endpoints, radii


def transform_capsule_array(msg, target_frame, tf_buffer):
    """Return a validated copy of ``msg`` expressed in ``target_frame``."""
    endpoints, radii = capsule_arrays(msg, target_frame, tf_buffer)

    result = CapsuleArray()
    result.header.stamp = msg.header.stamp
    result.header.frame_id = str(target_frame).strip()

    for source, (a, b), radius in zip(msg.capsules, endpoints, radii):
        capsule = Capsule()
        capsule.a.x, capsule.a.y, capsule.a.z = a
        capsule.b.x, capsule.b.y, capsule.b.z = b
        capsule.radius = float(radius)
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

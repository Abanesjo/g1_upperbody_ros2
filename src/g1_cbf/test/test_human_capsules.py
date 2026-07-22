import math

import pytest
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from tf2_ros import Buffer, TransformException

from g1_cbf.human_capsules import (
    human_data_is_fresh,
    human_data_time,
    transform_capsule_array,
)
from g1_cbf_msg.msg import Capsule, CapsuleArray


def _message(
    frame='world',
    stamp_sec=2,
    a=(1.0, 0.0, 0.0),
    b=(0.0, 1.0, 0.0),
    radius=0.25,
):
    msg = CapsuleArray()
    msg.header.frame_id = frame
    msg.header.stamp.sec = stamp_sec
    capsule = Capsule()
    capsule.a.x, capsule.a.y, capsule.a.z = a
    capsule.b.x, capsule.b.y, capsule.b.z = b
    capsule.radius = radius
    capsule.name = 'torso'
    msg.capsules.append(capsule)
    return msg


def _set_transform(buffer, stamp_sec, x, qz, qw):
    transform = TransformStamped()
    transform.header.frame_id = 'world'
    transform.child_frame_id = 'camera_optical'
    transform.header.stamp.sec = stamp_sec
    transform.transform.translation.x = x
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    buffer.set_transform(transform, 'pytest')


def test_world_identity_does_not_require_tf():
    result = transform_capsule_array(_message(), 'world', Buffer())

    assert result.header.frame_id == 'world'
    assert result.header.stamp.sec == 2
    assert result.capsules[0].a.x == 1.0
    assert result.capsules[0].b.y == 1.0
    assert result.capsules[0].radius == 0.25
    assert result.capsules[0].name == 'torso'


def test_camera_frame_uses_transform_at_message_stamp():
    buffer = Buffer()
    _set_transform(buffer, 1, x=1.0, qz=0.0, qw=1.0)
    _set_transform(buffer, 3, x=3.0, qz=1.0, qw=0.0)

    result = transform_capsule_array(
        _message(frame='camera_optical'),
        'world',
        buffer,
    )

    capsule = result.capsules[0]
    assert (capsule.a.x, capsule.a.y, capsule.a.z) == pytest.approx(
        (2.0, 1.0, 0.0), abs=1e-12,
    )
    assert (capsule.b.x, capsule.b.y, capsule.b.z) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-12,
    )


def test_zero_stamp_uses_latest_transform():
    buffer = Buffer()
    _set_transform(buffer, 1, x=1.0, qz=0.0, qw=1.0)
    _set_transform(buffer, 3, x=3.0, qz=1.0, qw=0.0)

    result = transform_capsule_array(
        _message(frame='camera_optical', stamp_sec=0),
        'world',
        buffer,
    )

    capsule = result.capsules[0]
    assert (capsule.a.x, capsule.a.y, capsule.a.z) == pytest.approx(
        (2.0, 0.0, 0.0), abs=1e-12,
    )
    assert (capsule.b.x, capsule.b.y, capsule.b.z) == pytest.approx(
        (3.0, -1.0, 0.0), abs=1e-12,
    )


def test_missing_transform_is_rejected():
    with pytest.raises(TransformException):
        transform_capsule_array(
            _message(frame='unconnected_camera'),
            'world',
            Buffer(),
        )


@pytest.mark.parametrize(
    ('frame', 'a_x', 'radius'),
    [
        ('', 0.0, 0.25),
        ('world', math.nan, 0.25),
        ('world', 0.0, 0.0),
        ('world', 0.0, math.inf),
    ],
)
def test_invalid_input_is_rejected(frame, a_x, radius):
    msg = _message(frame=frame, a=(a_x, 0.0, 0.0), radius=radius)
    with pytest.raises(ValueError):
        transform_capsule_array(msg, 'world', Buffer())


def test_human_timeout_is_inclusive():
    last_update = Time(seconds=10.0)

    assert human_data_is_fresh(last_update, Time(seconds=10.5), 0.5)
    assert not human_data_is_fresh(
        last_update,
        Time(seconds=10.500001),
        0.5,
    )
    assert not human_data_is_fresh(
        last_update,
        Time(seconds=9.0),
        0.5,
    )


def test_zero_measurement_time_uses_receipt_time():
    msg = _message(stamp_sec=0)
    receipt_time = Time(seconds=12.0)

    assert human_data_time(msg.header.stamp, receipt_time) == receipt_time


def test_empty_array_is_valid_without_tf_lookup():
    msg = CapsuleArray()
    msg.header.frame_id = 'camera_optical'

    result = transform_capsule_array(msg, 'world', Buffer())

    assert result.header.frame_id == 'world'
    assert not result.capsules

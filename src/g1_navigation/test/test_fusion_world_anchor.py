import math
from types import SimpleNamespace

from geometry_msgs.msg import PoseStamped, TransformStamped
import pytest
from tf2_ros import TransformException

import g1_navigation.fusion_world_anchor as anchor_module
from g1_navigation.fusion_world_anchor import FusionWorldAnchorNode
from g1_navigation.fusion_world_anchor import PoseWindow
from g1_navigation.fusion_world_anchor import anchor_pose
from g1_navigation.fusion_world_anchor import average_poses
from g1_navigation.fusion_world_anchor import pose_compose
from g1_navigation.fusion_world_anchor import pose_from_transform
from g1_navigation.fusion_world_anchor import pose_inverse
from g1_navigation.fusion_world_anchor import position_spread
from g1_navigation.fusion_world_anchor import quaternion_from_rpy
from g1_navigation.fusion_world_anchor import rpy_from_quaternion
from g1_navigation.fusion_world_anchor import tilt_deg


IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
# A ROS z-up frame -> camera optical frame (z forward, x right, y down).
OPTICAL_IN_BODY = quaternion_from_rpy(-math.pi / 2.0, 0.0, -math.pi / 2.0)


def _assert_pose_close(actual, expected, tolerance=1.0e-9):
    assert actual[0] == pytest.approx(expected[0], abs=tolerance)
    # q and -q are the same rotation.
    dot = sum(a * b for a, b in zip(actual[1], expected[1]))
    assert abs(dot) == pytest.approx(1.0, abs=tolerance)


def _pose_stamped(pose, frame='world'):
    message = PoseStamped()
    message.header.stamp.sec = 5
    message.header.frame_id = frame
    (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
    ) = pose[0]
    (
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    ) = pose[1]
    return message


def _transform(pose, parent='fusion_world', child='cam_optical'):
    message = TransformStamped()
    message.header.frame_id = parent
    message.child_frame_id = child
    (
        message.transform.translation.x,
        message.transform.translation.y,
        message.transform.translation.z,
    ) = pose[0]
    (
        message.transform.rotation.x,
        message.transform.rotation.y,
        message.transform.rotation.z,
        message.transform.rotation.w,
    ) = pose[1]
    return message


def test_rpy_round_trip_matches_static_transform_publisher_convention():
    # Pure yaw of 90 deg maps x onto y, as tf2 does.
    quaternion = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)
    assert quaternion == pytest.approx(
        (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    rpy = (0.1, -0.2, 2.5)
    assert rpy_from_quaternion(quaternion_from_rpy(*rpy)) == pytest.approx(rpy)


def test_pose_inverse_composes_to_identity():
    pose = ((1.0, -2.0, 0.5), quaternion_from_rpy(0.3, -0.1, 1.2))
    _assert_pose_close(pose_compose(pose, pose_inverse(pose)), IDENTITY)
    _assert_pose_close(pose_compose(pose_inverse(pose), pose), IDENTITY)


def test_anchor_recovers_world_to_fusion():
    # Ground truth: fusion_world sits 2 m ahead of world, yawed 30 deg.
    world_to_fusion = ((2.0, 0.5, 0.0), quaternion_from_rpy(0.0, 0.0, 0.52))
    # Camera 0.94 m above the fusion origin, pitched down 15 deg.
    fusion_to_optical = pose_compose(
        ((0.0, 0.0, 0.94), quaternion_from_rpy(0.0, 0.26, 0.0)),
        ((0.0, 0.0, 0.0), OPTICAL_IN_BODY),
    )
    subject_to_optical = ((0.01, -0.03, 0.0), quaternion_from_rpy(0.0, 0.02, 0.0))
    # What Vicon would report for the subject.
    world_to_subject = pose_compose(
        pose_compose(world_to_fusion, fusion_to_optical),
        pose_inverse(subject_to_optical),
    )

    result = anchor_pose(world_to_subject, subject_to_optical, fusion_to_optical)

    _assert_pose_close(result, world_to_fusion)
    assert tilt_deg(result[1]) == pytest.approx(0.0, abs=1.0e-6)


def test_identity_mount_means_subject_is_the_optical_frame():
    world_to_optical = ((1.0, 2.0, 1.4), OPTICAL_IN_BODY)
    fusion_to_optical = ((0.0, 0.0, 1.4), OPTICAL_IN_BODY)
    result = anchor_pose(world_to_optical, IDENTITY, fusion_to_optical)
    _assert_pose_close(result, ((1.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)))


def test_average_poses_handles_quaternion_sign_flips():
    quaternion = quaternion_from_rpy(0.0, 0.0, 0.4)
    flipped = tuple(-value for value in quaternion)
    poses = [
        ((1.0, 0.0, 0.0), quaternion),
        ((3.0, 0.0, 0.0), flipped),
    ]
    _assert_pose_close(average_poses(poses), ((2.0, 0.0, 0.0), quaternion))
    with pytest.raises(ValueError):
        average_poses([])


def test_position_spread_is_largest_distance_from_mean():
    poses = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        ((0.004, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    ]
    assert position_spread(poses) == pytest.approx(0.002)


def test_pose_window_keeps_the_newest_samples():
    window = PoseWindow(2)
    for x in (1.0, 2.0, 3.0):
        window.add(((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    assert window.full
    assert window.average()[0] == pytest.approx((2.5, 0.0, 0.0))
    with pytest.raises(ValueError):
        PoseWindow(0)


def test_pose_from_transform_rejects_non_finite_values():
    with pytest.raises(ValueError):
        pose_from_transform(_transform(((math.nan, 0.0, 0.0), IDENTITY[1])))


class _FakeBuffer:
    def __init__(self, transform=None):
        self.transform = transform
        self.lookups = []

    def lookup_transform(self, target, source, time):
        self.lookups.append((target, source))
        if self.transform is None:
            raise TransformException('not yet')
        return self.transform


def _node(sample_count=3, buffer=None, max_spread=0.01):
    node = FusionWorldAnchorNode.__new__(FusionWorldAnchorNode)
    node._world_frame = 'world'
    node._fusion_frame = 'fusion_world'
    node._camera_frame = 'cam_optical'
    node._camera_pose_topic = '/vicon/cam/cam'
    node._sample_count = sample_count
    node._max_position_spread_m = max_spread
    node._max_tilt_warn_deg = 3.0
    node._subject_to_optical = IDENTITY
    node._window = PoseWindow(sample_count)
    node._published = False
    node._tf_buffer = buffer or _FakeBuffer()
    node.sent = []
    node.warnings = []
    node.destroyed = []
    node._static_broadcaster = SimpleNamespace(sendTransform=node.sent.append)
    node._pose_subscription = 'subscription'
    node._timer = SimpleNamespace(cancel=lambda: node.destroyed.append('timer'))
    node.destroy_subscription = node.destroyed.append
    node._warn = node.warnings.append
    logger = SimpleNamespace(info=lambda message: None,
                             warn=node.warnings.append)
    node.get_logger = lambda: logger
    stamp = SimpleNamespace(to_msg=lambda: _pose_stamped(IDENTITY).header.stamp)
    node.get_clock = lambda: SimpleNamespace(now=lambda: stamp)
    return node


def test_callback_skips_occluded_and_wrong_frame_poses():
    node = _node()
    node._pose_callback(_pose_stamped(((math.nan, 0.0, 0.0), IDENTITY[1])))
    node._pose_callback(_pose_stamped(IDENTITY, frame='vicon'))
    node._pose_callback(_pose_stamped(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))))
    assert node._window.size == 0
    assert len(node.warnings) == 3
    node._pose_callback(_pose_stamped(IDENTITY))
    assert node._window.size == 1


def test_waits_for_samples_then_fusion_tf_then_publishes_once():
    world_to_optical = ((1.0, 2.0, 1.4), OPTICAL_IN_BODY)
    buffer = _FakeBuffer()
    node = _node(buffer=buffer)

    node._try_publish()
    assert node.sent == [] and buffer.lookups == []

    for _ in range(3):
        node._pose_callback(_pose_stamped(world_to_optical))
    node._try_publish()
    assert node.sent == []
    assert buffer.lookups == [('fusion_world', 'cam_optical')]

    buffer.transform = _transform(((0.0, 0.0, 1.4), OPTICAL_IN_BODY))
    node._try_publish()
    assert len(node.sent) == 1
    transform = node.sent[0]
    assert transform.header.frame_id == 'world'
    assert transform.child_frame_id == 'fusion_world'
    _assert_pose_close(
        pose_from_transform(transform),
        ((1.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert node.destroyed == ['subscription', 'timer']

    node._try_publish()
    assert len(node.sent) == 1


def test_moving_camera_is_not_anchored():
    buffer = _FakeBuffer(_transform(IDENTITY))
    node = _node(buffer=buffer, max_spread=0.01)
    for x in (0.0, 0.05, 0.1):
        node._pose_callback(_pose_stamped(((x, 0.0, 0.0), IDENTITY[1])))
    node._try_publish()
    assert node.sent == []
    assert 'moving' in node.warnings[-1]


def test_large_tilt_is_reported():
    tilted = ((0.0, 0.0, 0.0), quaternion_from_rpy(0.2, 0.0, 0.0))
    node = _node(buffer=_FakeBuffer(_transform(IDENTITY)))
    for _ in range(3):
        node._pose_callback(_pose_stamped(tilted))
    node._try_publish()
    assert len(node.sent) == 1
    assert 'tilted' in node.warnings[-1]


def test_module_exposes_a_main_entry_point():
    assert callable(anchor_module.main)

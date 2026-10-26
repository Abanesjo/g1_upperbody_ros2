import math

import pytest
from visualization_msgs.msg import Marker
from zed_msgs.msg import Object, ObjectsStamped

from g1_human.skeleton_markers import (
    BODY_18,
    BODY_34,
    BODY_38,
    BONES,
    KEYPOINT_COUNTS,
    MARKER_NAMESPACE_BONES,
    MARKER_NAMESPACE_JOINTS,
    color_for_id,
    skeleton_markers,
    valid_keypoints,
)


def _object(body_format, body_id=3, missing=()):
    obj = Object()
    obj.label_id = body_id
    obj.body_format = body_format
    obj.skeleton_available = True
    for index in range(KEYPOINT_COUNTS[body_format]):
        value = math.nan if index in missing else float(index)
        obj.skeleton_3d.keypoints[index].kp = [value, 0.5 * value, 1.0]
    return obj


def _message(*objects):
    msg = ObjectsStamped()
    msg.header.frame_id = 'fusion_world'
    msg.header.stamp.sec = 12
    msg.header.stamp.nanosec = 345
    msg.objects = list(objects)
    return msg


@pytest.mark.parametrize('body_format', [BODY_18, BODY_34, BODY_38])
def test_bone_tables_index_existing_keypoints(body_format):
    count = KEYPOINT_COUNTS[body_format]
    bones = BONES[body_format]
    assert len(set(bones)) == len(bones)
    for first, second in bones:
        assert 0 <= first < count
        assert 0 <= second < count
        assert first != second
    # Every keypoint is drawn by at least one bone.
    assert {index for bone in bones for index in bone} == set(range(count))


def test_valid_keypoints_reads_only_the_format_slots():
    obj = _object(BODY_18)
    # The message has 70 slots; slots past the format's count are ignored.
    obj.skeleton_3d.keypoints[40].kp = [9.0, 9.0, 9.0]
    assert sorted(valid_keypoints(obj)) == list(range(18))


def test_valid_keypoints_rejects_unusable_objects():
    no_skeleton = _object(BODY_18)
    no_skeleton.skeleton_available = False
    unknown_format = _object(BODY_18)
    unknown_format.body_format = 7
    assert valid_keypoints(no_skeleton) == {}
    assert valid_keypoints(unknown_format) == {}


def test_markers_keep_the_skeleton_header():
    markers = skeleton_markers(_message(_object(BODY_34))).markers
    assert [marker.action for marker in markers] == [
        Marker.DELETEALL, Marker.ADD, Marker.ADD]
    for marker in markers:
        assert marker.header.frame_id == 'fusion_world'
        assert (marker.header.stamp.sec, marker.header.stamp.nanosec) == (
            12, 345)


def test_nan_keypoints_and_their_bones_are_skipped():
    # BODY_18 keypoint 4 (right wrist) appears only in bone (3, 4).
    msg = _message(_object(BODY_18, missing={4}))
    _, joints, bones = skeleton_markers(msg, lifetime_sec=0.25).markers

    assert joints.ns == MARKER_NAMESPACE_JOINTS
    assert joints.type == Marker.SPHERE_LIST
    assert len(joints.points) == 17
    assert all(point.x != 4.0 for point in joints.points)

    assert bones.ns == MARKER_NAMESPACE_BONES
    assert bones.type == Marker.LINE_LIST
    assert len(bones.points) == 2 * (len(BONES[BODY_18]) - 1)
    assert (joints.lifetime.sec, joints.lifetime.nanosec) == (0, 250_000_000)


def test_each_body_gets_its_own_markers_and_colour():
    msg = _message(_object(BODY_18, body_id=0), _object(BODY_38, body_id=1))
    markers = skeleton_markers(msg).markers[1:]
    assert [(marker.ns, marker.id) for marker in markers] == [
        (MARKER_NAMESPACE_JOINTS, 0), (MARKER_NAMESPACE_BONES, 0),
        (MARKER_NAMESPACE_JOINTS, 1), (MARKER_NAMESPACE_BONES, 1),
    ]
    assert markers[0].color == color_for_id(0)
    assert markers[2].color == color_for_id(1)
    assert markers[0].color != markers[2].color


def test_empty_message_only_clears():
    markers = skeleton_markers(_message()).markers
    assert [marker.action for marker in markers] == [Marker.DELETEALL]

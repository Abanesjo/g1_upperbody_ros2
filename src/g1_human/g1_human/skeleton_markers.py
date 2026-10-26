"""RViz markers for ZED body-tracking skeletons (zed_msgs/ObjectsStamped).

Markers keep the skeleton message's header, so TF places them: with
vicon.launch.xml publishing world -> fusion_world, a skeleton in fusion_world
shows up in the world frame.
"""

import math

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# zed_msgs Object.body_format: the sl::BODY_FORMAT enum value.
BODY_18 = 0
BODY_34 = 1
BODY_38 = 2

KEYPOINT_COUNTS = {BODY_18: 18, BODY_34: 34, BODY_38: 38}

# Keypoint index pairs of sl::BODY_18_BONES, sl::BODY_34_BONES and
# sl::BODY_38_BONES (ZED SDK 5.3, sl/Camera.hpp), in the SDK's order.
BONES = {
    BODY_18: (
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (2, 8),
        (8, 9), (9, 10), (5, 11), (11, 12), (12, 13), (2, 5), (8, 11),
        (0, 14), (14, 16), (0, 15), (15, 17),
    ),
    BODY_34: (
        (0, 1), (1, 2), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (7, 10), (2, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
        (14, 17), (0, 18), (18, 19), (19, 20), (20, 21), (0, 22), (22, 23),
        (23, 24), (24, 25), (2, 3), (3, 26), (26, 27), (27, 28), (28, 29),
        (27, 30), (30, 31), (20, 32), (24, 33), (32, 21), (33, 25),
    ),
    BODY_38: (
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (5, 7),
        (7, 9), (3, 10), (10, 12), (12, 14), (14, 16), (16, 30), (16, 32),
        (16, 34), (16, 36), (3, 11), (11, 13), (13, 15), (15, 17), (17, 31),
        (17, 33), (17, 35), (17, 37), (0, 18), (18, 20), (20, 22), (22, 28),
        (22, 24), (22, 26), (0, 19), (19, 21), (21, 23), (23, 29), (23, 25),
        (23, 27),
    ),
}

# Same palette as zedx_driver_ros2's skeleton overlay (skeleton_draw_utils.hpp),
# converted from OpenCV BGR to RGB.
PALETTE = (
    (59, 176, 232), (25, 208, 175), (105, 205, 102), (255, 0, 185),
    (252, 107, 99), (8, 225, 252), (141, 130, 167), (113, 72, 194),
)
UNTRACKED_RGB = (36, 184, 236)

MARKER_NAMESPACE_JOINTS = 'skeleton_joints'
MARKER_NAMESPACE_BONES = 'skeleton_bones'


def color_for_id(body_id: int, alpha: float = 1.0) -> ColorRGBA:
    """Stable colour per tracking id; negative ids are untracked bodies."""
    rgb = UNTRACKED_RGB if body_id < 0 else PALETTE[body_id % len(PALETTE)]
    return ColorRGBA(
        r=rgb[0] / 255.0, g=rgb[1] / 255.0, b=rgb[2] / 255.0, a=alpha)


def valid_keypoints(obj):
    """Finite 3D keypoints of one object as {index: Point}.

    Returns an empty dict when the object has no skeleton or an unknown
    body_format. Only the first 18/34/38 of the 70 slots are read; the ZED SDK
    marks keypoints it did not detect as NaN.
    """
    if not obj.skeleton_available:
        return {}
    count = KEYPOINT_COUNTS.get(int(obj.body_format))
    if count is None:
        return {}
    points = {}
    for index, keypoint in enumerate(obj.skeleton_3d.keypoints[:count]):
        x, y, z = (float(value) for value in keypoint.kp)
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            points[index] = Point(x=x, y=y, z=z)
    return points


def skeleton_markers(
    msg,
    joint_radius: float = 0.04,
    bone_width: float = 0.02,
    lifetime_sec: float = 0.5,
) -> MarkerArray:
    """One SPHERE_LIST (joints) and one LINE_LIST (bones) per skeleton.

    The array starts with a DELETEALL so bodies that left the scene disappear
    on the next message instead of waiting for their lifetime.
    """
    markers = MarkerArray()
    clear = Marker()
    clear.header = msg.header
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    lifetime_ns = int(round(max(lifetime_sec, 0.0) * 1e9))
    for marker_id, obj in enumerate(msg.objects):
        points = valid_keypoints(obj)
        if not points:
            continue
        color = color_for_id(int(obj.label_id))

        joints = Marker()
        joints.header = msg.header
        joints.ns = MARKER_NAMESPACE_JOINTS
        joints.id = marker_id
        joints.type = Marker.SPHERE_LIST
        joints.action = Marker.ADD
        joints.pose.orientation.w = 1.0
        joints.scale.x = joints.scale.y = joints.scale.z = 2.0 * joint_radius
        joints.color = color
        joints.points = [points[index] for index in sorted(points)]
        joints.lifetime.sec = lifetime_ns // 1_000_000_000
        joints.lifetime.nanosec = lifetime_ns % 1_000_000_000
        markers.markers.append(joints)

        bones = Marker()
        bones.header = msg.header
        bones.ns = MARKER_NAMESPACE_BONES
        bones.id = marker_id
        bones.type = Marker.LINE_LIST
        bones.action = Marker.ADD
        bones.pose.orientation.w = 1.0
        bones.scale.x = bone_width
        bones.color = color
        for first, second in BONES[int(obj.body_format)]:
            if first in points and second in points:
                bones.points.extend((points[first], points[second]))
        bones.lifetime = joints.lifetime
        if bones.points:
            markers.markers.append(bones)

    return markers

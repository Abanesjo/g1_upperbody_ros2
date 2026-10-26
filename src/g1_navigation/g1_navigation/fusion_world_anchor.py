"""
Anchor the ZED fusion frame in the Vicon world from one tracked camera.

zedx_driver_ros2 publishes skeletons in ``fusion_world`` and the static TF
``fusion_world -> <camera>_left_camera_optical_frame`` for every camera.
Vicon tracks the same camera as a subject whose origin and axes are the
camera's optical frame, up to the fixed ``mount`` offset.  Averaging the Vicon
pose over a short window and composing

    world <- fusion_world = (world <- subject) (subject <- optical)
                            (fusion_world <- optical)^-1

gives one static transform ``world -> fusion_world``.  A TF frame has only one
parent, so the camera frames stay children of ``fusion_world``; the camera is
only used to find where ``fusion_world`` sits.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Deque, Optional, Sequence, Tuple

from geometry_msgs.msg import PoseStamped, TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException
from tf2_ros import TransformListener

from g1_navigation.apriltag_localization import (
    _quaternion,
    _vector,
    finite_nonnegative,
    finite_positive,
    normalize_frame_id,
    normalize_quaternion,
    normalized_pose,
    positive_integer,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    vector_norm,
)


DEFAULT_CAMERA_POSE_TOPIC = '/vicon/zed_10025986/zed_10025986'
DEFAULT_WORLD_FRAME = 'world'
DEFAULT_FUSION_FRAME = 'fusion_world'
DEFAULT_CAMERA_FRAME = 'zed_right_left_camera_optical_frame'
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_MAX_POSITION_SPREAD_M = 0.01
DEFAULT_MAX_TILT_WARN_DEG = 3.0
DEFAULT_RETRY_PERIOD_SEC = 1.0
WARNING_THROTTLE_SEC = 5.0

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
Pose = Tuple[Vector3, Quaternion]


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Fixed-axis roll, pitch, yaw, as static_transform_publisher reads them."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def rpy_from_quaternion(quaternion: Quaternion) -> Vector3:
    """Inverse of :func:`quaternion_from_rpy`."""
    x, y, z, w = normalize_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def pose_compose(lhs: Pose, rhs: Pose) -> Pose:
    """Return ``a <- c`` from ``a <- b`` and ``b <- c``."""
    lhs_translation, lhs_quaternion = lhs
    rhs_translation, rhs_quaternion = rhs
    rotated = rotate_vector(lhs_quaternion, rhs_translation)
    translation = tuple(
        lhs_value + rotated_value
        for lhs_value, rotated_value in zip(lhs_translation, rotated)
    )
    quaternion = normalize_quaternion(
        quaternion_multiply(lhs_quaternion, rhs_quaternion)
    )
    return (translation, quaternion)


def pose_inverse(pose: Pose) -> Pose:
    """Return ``b <- a`` from ``a <- b``."""
    translation, quaternion = pose
    inverse_quaternion = quaternion_conjugate(normalize_quaternion(quaternion))
    rotated = rotate_vector(inverse_quaternion, translation)
    return (tuple(-value for value in rotated), inverse_quaternion)


def anchor_pose(
    world_to_subject: Pose,
    subject_to_optical: Pose,
    fusion_to_optical: Pose,
) -> Pose:
    """Return ``world <- fusion`` from the three measured/configured poses."""
    world_to_optical = pose_compose(world_to_subject, subject_to_optical)
    return pose_compose(world_to_optical, pose_inverse(fusion_to_optical))


def average_poses(poses: Sequence[Pose]) -> Pose:
    """Mean translation and sign-aligned, normalized mean quaternion.

    Adequate for samples of one static object, which differ by noise only.
    """
    if not poses:
        raise ValueError('cannot average zero poses')
    count = float(len(poses))
    translation = tuple(
        sum(pose[0][axis] for pose in poses) / count for axis in range(3)
    )
    reference = poses[0][1]
    summed = [0.0, 0.0, 0.0, 0.0]
    for _, quaternion in poses:
        sign = (
            -1.0
            if sum(a * b for a, b in zip(reference, quaternion)) < 0.0
            else 1.0
        )
        for axis in range(4):
            summed[axis] += sign * quaternion[axis]
    return (translation, normalize_quaternion(summed))


def position_spread(poses: Sequence[Pose]) -> float:
    """Largest distance of a sample position from the mean position."""
    if not poses:
        return 0.0
    mean = average_poses(poses)[0]
    return max(
        vector_norm(tuple(a - b for a, b in zip(pose[0], mean)))
        for pose in poses
    )


def tilt_deg(quaternion: Quaternion) -> float:
    """Angle between a frame's z axis and its parent's z axis, in degrees."""
    z_axis = rotate_vector(normalize_quaternion(quaternion), (0.0, 0.0, 1.0))
    return math.degrees(math.acos(max(-1.0, min(1.0, z_axis[2]))))


def pose_from_pose_stamped(message: PoseStamped) -> Pose:
    return (_vector(message.pose.position), _quaternion(message.pose.orientation))


def pose_from_transform(message: TransformStamped) -> Pose:
    translation = _vector(message.transform.translation)
    if not all(math.isfinite(value) for value in translation):
        raise ValueError('transform translation contains non-finite values')
    return (
        translation,
        normalize_quaternion(_quaternion(message.transform.rotation)),
    )


def pose_to_transform(
    pose: Pose,
    parent_frame: str,
    child_frame: str,
) -> TransformStamped:
    result = TransformStamped()
    result.header.frame_id = parent_frame
    result.child_frame_id = child_frame
    (
        result.transform.translation.x,
        result.transform.translation.y,
        result.transform.translation.z,
    ) = pose[0]
    (
        result.transform.rotation.x,
        result.transform.rotation.y,
        result.transform.rotation.z,
        result.transform.rotation.w,
    ) = pose[1]
    return result


class PoseWindow:
    """Rolling window of the newest valid camera poses."""

    def __init__(self, sample_count: int):
        self.sample_count = positive_integer(sample_count, 'sample_count')
        self._poses: Deque[Pose] = deque(maxlen=self.sample_count)

    def add(self, pose: Pose):
        self._poses.append(pose)

    @property
    def size(self) -> int:
        return len(self._poses)

    @property
    def full(self) -> bool:
        return len(self._poses) == self.sample_count

    def spread(self) -> float:
        return position_spread(list(self._poses))

    def average(self) -> Pose:
        return average_poses(list(self._poses))


class FusionWorldAnchorNode(Node):
    """Publish a static ``world -> fusion_world`` from a Vicon-tracked camera."""

    def __init__(self):
        super().__init__('fusion_world_anchor_node')
        self._declare_parameters()
        self._read_parameters()

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._window = PoseWindow(self._sample_count)
        self._published = False
        self._pose_subscription = self.create_subscription(
            PoseStamped,
            self._camera_pose_topic,
            self._pose_callback,
            input_qos,
        )
        self._timer = self.create_timer(
            DEFAULT_RETRY_PERIOD_SEC,
            self._try_publish,
        )

        self.get_logger().info(
            f'Anchoring {self._world_frame} -> {self._fusion_frame} from '
            f'{self._camera_pose_topic} ({self._camera_frame}), averaging '
            f'{self._sample_count} samples'
        )

    def _declare_parameters(self):
        defaults = {
            'camera_pose_topic': DEFAULT_CAMERA_POSE_TOPIC,
            'world_frame': DEFAULT_WORLD_FRAME,
            'fusion_frame': DEFAULT_FUSION_FRAME,
            'camera_frame': DEFAULT_CAMERA_FRAME,
            'sample_count': DEFAULT_SAMPLE_COUNT,
            'max_position_spread_m': DEFAULT_MAX_POSITION_SPREAD_M,
            'max_tilt_warn_deg': DEFAULT_MAX_TILT_WARN_DEG,
            'mount_x': 0.0,
            'mount_y': 0.0,
            'mount_z': 0.0,
            'mount_roll': 0.0,
            'mount_pitch': 0.0,
            'mount_yaw': 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        def get(name):
            return self.get_parameter(name).value

        self._world_frame = normalize_frame_id(
            get('world_frame'), 'world_frame')
        self._fusion_frame = normalize_frame_id(
            get('fusion_frame'), 'fusion_frame')
        self._camera_frame = normalize_frame_id(
            get('camera_frame'), 'camera_frame')
        frames = (self._world_frame, self._fusion_frame, self._camera_frame)
        if len(set(frames)) != len(frames):
            raise ValueError('anchor frame parameters must be distinct')
        self._camera_pose_topic = str(get('camera_pose_topic')).strip()
        if not self._camera_pose_topic:
            raise ValueError('camera_pose_topic must not be empty')
        self._sample_count = positive_integer(
            get('sample_count'), 'sample_count')
        self._max_position_spread_m = finite_positive(
            get('max_position_spread_m'), 'max_position_spread_m')
        self._max_tilt_warn_deg = finite_nonnegative(
            get('max_tilt_warn_deg'), 'max_tilt_warn_deg')

        mount = [float(get(f'mount_{name}')) for name in (
            'x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        if not all(math.isfinite(value) for value in mount):
            raise ValueError('mount_* parameters must be finite')
        self._subject_to_optical = (
            tuple(mount[:3]),
            quaternion_from_rpy(*mount[3:]),
        )

    def _pose_callback(self, message: PoseStamped):
        # Occluded subjects arrive as NaN poses; skip them.
        try:
            pose = normalized_pose(message, self._world_frame)
        except ValueError as exc:
            self._warn(f'ignoring camera pose: {exc}')
            return
        self._window.add(pose_from_pose_stamped(pose))

    def _try_publish(self):
        if self._published:
            return
        if not self._window.full:
            self._warn(
                f'waiting for {self._camera_pose_topic}: '
                f'{self._window.size}/{self._sample_count} valid samples'
            )
            return
        spread = self._window.spread()
        if spread > self._max_position_spread_m:
            self._warn(
                f'camera pose spread {spread * 1000.0:.1f} mm exceeds '
                f'{self._max_position_spread_m * 1000.0:.1f} mm; the camera or '
                'its markers are moving'
            )
            return
        try:
            fusion_to_optical = pose_from_transform(
                self._tf_buffer.lookup_transform(
                    self._fusion_frame,
                    self._camera_frame,
                    Time(),
                )
            )
        except (TransformException, ValueError) as exc:
            self._warn(
                f'waiting for TF {self._fusion_frame} -> '
                f'{self._camera_frame} from the ZED fusion node: {exc}'
            )
            return

        world_to_fusion = anchor_pose(
            self._window.average(),
            self._subject_to_optical,
            fusion_to_optical,
        )
        transform = pose_to_transform(
            world_to_fusion, self._world_frame, self._fusion_frame)
        transform.header.stamp = self.get_clock().now().to_msg()
        self._static_broadcaster.sendTransform(transform)
        self._published = True
        # The static TF stays latched while the node runs; no more input needed.
        self.destroy_subscription(self._pose_subscription)
        self._timer.cancel()

        translation, quaternion = world_to_fusion
        roll, pitch, yaw = (
            math.degrees(value) for value in rpy_from_quaternion(quaternion))
        tilt = tilt_deg(quaternion)
        self.get_logger().info(
            f'Published static TF {self._world_frame} -> {self._fusion_frame}: '
            f'xyz [{translation[0]:.3f}, {translation[1]:.3f}, '
            f'{translation[2]:.3f}] m, rpy [{roll:.2f}, {pitch:.2f}, '
            f'{yaw:.2f}] deg, tilt {tilt:.2f} deg, sample spread '
            f'{spread * 1000.0:.1f} mm'
        )
        if tilt > self._max_tilt_warn_deg:
            self.get_logger().warn(
                f'{self._fusion_frame} is tilted {tilt:.2f} deg from '
                f'{self._world_frame}. Both should be gravity-aligned; check '
                'the Vicon subject axes of the camera and the mount_* offset.'
            )

    def _warn(self, message: str):
        self.get_logger().warn(
            message,
            throttle_duration_sec=WARNING_THROTTLE_SEC,
        )


def main(args=None):
    """Run the fusion-world anchor node until ROS shuts down."""
    rclpy.init(args=args)
    node = FusionWorldAnchorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

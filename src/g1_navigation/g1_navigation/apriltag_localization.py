"""
Localize the pelvis from a stamped, externally detected AprilTag pose.

The detected ``tag_frame`` and the URDF's ``tag_mount_frame`` describe the
same physical marker, but they intentionally remain distinct TF frames.  For
every tag measurement this node evaluates the articulated robot kinematics at
the *same timestamp* and composes

    world <- pelvis = (world <- tag_frame) (tag_mount_frame <- pelvis).

No latest-TF fallback or level-pelvis assumption is used.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional, Sequence, Tuple

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException
from tf2_ros import TransformListener


DEFAULT_TAG_POSE_TOPIC = '/pose/tag_frame'
DEFAULT_PELVIS_POSE_TOPIC = '/pose/pelvis'
DEFAULT_ODOM_TOPIC = '/odom'
DEFAULT_DIAGNOSTICS_TOPIC = '/diagnostics'
DEFAULT_WORLD_FRAME = 'world'
DEFAULT_DETECTED_TAG_FRAME = 'tag_frame'
DEFAULT_TAG_MOUNT_FRAME = 'tag_mount_frame'
DEFAULT_PELVIS_FRAME = 'pelvis'
DEFAULT_QUEUE_DEPTH = 20
DEFAULT_KINEMATIC_WAIT_TIMEOUT_SEC = 0.20
DEFAULT_SOURCE_STALE_TIMEOUT_SEC = 0.50
DEFAULT_FUTURE_TOLERANCE_SEC = 0.05
DEFAULT_TF_LOOKUP_TIMEOUT_SEC = 0.0
DEFAULT_RETRY_RATE_HZ = 100.0
DEFAULT_DIAGNOSTIC_RATE_HZ = 2.0
DEFAULT_POSE_POSITION_STDDEV_M = 0.05
DEFAULT_POSE_ORIENTATION_STDDEV_RAD = math.radians(5.0)
DEFAULT_TWIST_LINEAR_STDDEV_MPS = 1.0
DEFAULT_TWIST_ANGULAR_STDDEV_RADPS = 1.0
DEFAULT_TWIST_MIN_DT_SEC = 0.02
DEFAULT_TWIST_MAX_DT_SEC = 0.50
DEFAULT_TWIST_MAX_LINEAR_SPEED_MPS = 3.0
DEFAULT_TWIST_MAX_ANGULAR_SPEED_RADPS = 4.0
DEFAULT_TWIST_MAX_TRANSLATION_STEP_M = 0.50
DEFAULT_TWIST_MAX_ROTATION_STEP_RAD = math.radians(45.0)
UNKNOWN_VARIANCE = 1.0e6
WARNING_THROTTLE_SEC = 2.0
MIN_QUATERNION_NORM = 1.0e-12

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
StampKey = Tuple[int, int]


@dataclass
class PendingPose:
    """A valid tag pose waiting for exact-time robot kinematics."""

    pose: PoseStamped
    stamp_nanoseconds: int
    enqueue_nanoseconds: int


@dataclass
class TwistEstimate:
    """Finite-difference twist and whether its covariance is meaningful."""

    twist: Twist
    valid: bool
    reason: str


def normalize_frame_id(value, parameter_name: str) -> str:
    """Return a canonical TF frame name or raise for an empty value."""
    frame_id = str(value or '').strip().lstrip('/')
    if not frame_id:
        raise ValueError(f'{parameter_name} must not be empty')
    return frame_id


def validate_frame_ids(
    world_frame,
    detected_tag_frame,
    tag_mount_frame,
    pelvis_frame,
) -> Tuple[str, str, str, str]:
    """Normalize and require all localization frames to be distinct."""
    frames = (
        normalize_frame_id(world_frame, 'world_frame'),
        normalize_frame_id(detected_tag_frame, 'detected_tag_frame'),
        normalize_frame_id(tag_mount_frame, 'tag_mount_frame'),
        normalize_frame_id(pelvis_frame, 'pelvis_frame'),
    )
    if len(set(frames)) != len(frames):
        raise ValueError('localization frame parameters must be distinct')
    return frames


def finite_positive(value, name: str) -> float:
    """Convert a parameter to a finite positive float."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f'{name} must be finite and positive')
    return result


def finite_nonnegative(value, name: str) -> float:
    """Convert a parameter to a finite non-negative float."""
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f'{name} must be finite and non-negative')
    return result


def positive_integer(value, name: str) -> int:
    """Convert a parameter to a positive integer without truncation."""
    result = int(value)
    if isinstance(value, float) and value != result:
        raise ValueError(f'{name} must be a positive integer')
    if result <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return result


def stamp_key(stamp) -> Optional[StampKey]:
    """Return a key for a nonzero ROS stamp, otherwise ``None``."""
    key = (int(stamp.sec), int(stamp.nanosec))
    return None if key == (0, 0) else key


def stamp_nanoseconds(stamp) -> Optional[int]:
    """Return a nonzero ROS stamp as integer nanoseconds."""
    key = stamp_key(stamp)
    if key is None:
        return None
    return key[0] * 1_000_000_000 + key[1]


def source_age_sec(stamp, now_nanoseconds: int) -> Optional[float]:
    """Return signed source age, or ``None`` for an invalid zero stamp."""
    value = stamp_nanoseconds(stamp)
    if value is None:
        return None
    return (int(now_nanoseconds) - value) * 1.0e-9


def _vector(message) -> Vector3:
    return (float(message.x), float(message.y), float(message.z))


def _quaternion(message) -> Quaternion:
    return (
        float(message.x),
        float(message.y),
        float(message.z),
        float(message.w),
    )


def normalize_quaternion(quaternion: Sequence[float]) -> Quaternion:
    """Return a finite unit quaternion."""
    result = tuple(float(value) for value in quaternion)
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError('quaternion contains non-finite values')
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= MIN_QUATERNION_NORM:
        raise ValueError('quaternion is degenerate')
    return tuple(value / norm for value in result)


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    """Return the conjugate of an xyzw quaternion."""
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def quaternion_multiply(
    lhs: Quaternion,
    rhs: Quaternion,
) -> Quaternion:
    """Hamilton product, with ROS xyzw component order."""
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector without constructing intermediate ROS messages."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quaternion_rotation_vector(quaternion: Quaternion) -> Vector3:
    """Map a unit quaternion to the shortest SO(3) rotation vector."""
    x, y, z, w = normalize_quaternion(quaternion)
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    vector_norm = math.sqrt(x * x + y * y + z * z)
    if vector_norm < 1.0e-12:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, w))
    scale = angle / vector_norm
    return (scale * x, scale * y, scale * z)


def vector_norm(vector: Vector3) -> float:
    """Return the Euclidean norm of a three-dimensional vector."""
    return math.sqrt(sum(value * value for value in vector))


def normalized_pose(
    source: PoseStamped,
    expected_world_frame: str,
) -> PoseStamped:
    """Validate a measured tag pose and return a normalized copy."""
    source_frame = normalize_frame_id(
        source.header.frame_id,
        'tag pose header.frame_id',
    )
    if source_frame != expected_world_frame:
        raise ValueError(
            f'tag pose frame is {source_frame}, expected '
            f'{expected_world_frame}'
        )
    translation = _vector(source.pose.position)
    if not all(math.isfinite(value) for value in translation):
        raise ValueError('tag pose position contains non-finite values')
    quaternion = normalize_quaternion(_quaternion(source.pose.orientation))

    result = PoseStamped()
    result.header.stamp = source.header.stamp
    result.header.frame_id = expected_world_frame
    (
        result.pose.position.x,
        result.pose.position.y,
        result.pose.position.z,
    ) = translation
    (
        result.pose.orientation.x,
        result.pose.orientation.y,
        result.pose.orientation.z,
        result.pose.orientation.w,
    ) = quaternion
    return result


def compose_pelvis_pose(
    tag_pose: PoseStamped,
    mount_to_pelvis: TransformStamped,
    *,
    world_frame: str,
    tag_mount_frame: str,
    pelvis_frame: str,
) -> PoseStamped:
    """Compose ``world <- tag`` with ``tag_mount <- pelvis``."""
    parent = normalize_frame_id(
        mount_to_pelvis.header.frame_id,
        'kinematic transform parent',
    )
    child = normalize_frame_id(
        mount_to_pelvis.child_frame_id,
        'kinematic transform child',
    )
    if parent != tag_mount_frame or child != pelvis_frame:
        raise ValueError(
            f'unexpected kinematic transform {parent} <- {child}'
        )

    tag_translation = _vector(tag_pose.pose.position)
    tag_quaternion = normalize_quaternion(
        _quaternion(tag_pose.pose.orientation)
    )
    mount_translation = _vector(mount_to_pelvis.transform.translation)
    if not all(math.isfinite(value) for value in mount_translation):
        raise ValueError('kinematic translation contains non-finite values')
    mount_quaternion = normalize_quaternion(
        _quaternion(mount_to_pelvis.transform.rotation)
    )

    rotated_mount = rotate_vector(tag_quaternion, mount_translation)
    pelvis_translation = tuple(
        tag_value + mount_value
        for tag_value, mount_value in zip(tag_translation, rotated_mount)
    )
    pelvis_quaternion = normalize_quaternion(
        quaternion_multiply(tag_quaternion, mount_quaternion)
    )

    result = PoseStamped()
    result.header.stamp = tag_pose.header.stamp
    result.header.frame_id = world_frame
    (
        result.pose.position.x,
        result.pose.position.y,
        result.pose.position.z,
    ) = pelvis_translation
    (
        result.pose.orientation.x,
        result.pose.orientation.y,
        result.pose.orientation.z,
        result.pose.orientation.w,
    ) = pelvis_quaternion
    return result


def estimate_child_twist(
    previous: Optional[PoseStamped],
    current: PoseStamped,
    *,
    min_dt_sec: float,
    max_dt_sec: float,
    max_linear_speed_mps: float,
    max_angular_speed_radps: float,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> TwistEstimate:
    """Estimate an unfiltered child-frame twist from two SE(3) poses."""
    twist = Twist()
    if previous is None:
        return TwistEstimate(twist, False, 'no previous pose')

    previous_stamp = stamp_nanoseconds(previous.header.stamp)
    current_stamp = stamp_nanoseconds(current.header.stamp)
    if previous_stamp is None or current_stamp is None:
        return TwistEstimate(twist, False, 'zero pose timestamp')
    dt_sec = (current_stamp - previous_stamp) * 1.0e-9
    if dt_sec < min_dt_sec or dt_sec > max_dt_sec:
        return TwistEstimate(twist, False, f'dt {dt_sec:.3f} s outside gate')

    previous_position = _vector(previous.pose.position)
    current_position = _vector(current.pose.position)
    translation_delta_world = tuple(
        current_value - previous_value
        for previous_value, current_value
        in zip(previous_position, current_position)
    )
    translation_step = vector_norm(translation_delta_world)
    if translation_step > max_translation_step_m:
        return TwistEstimate(
            twist,
            False,
            f'translation step {translation_step:.3f} m outside gate',
        )

    previous_quaternion = normalize_quaternion(
        _quaternion(previous.pose.orientation)
    )
    current_quaternion = normalize_quaternion(
        _quaternion(current.pose.orientation)
    )
    world_rotation_delta = normalize_quaternion(
        quaternion_multiply(
            current_quaternion,
            quaternion_conjugate(previous_quaternion),
        )
    )
    rotation_vector_world = quaternion_rotation_vector(
        world_rotation_delta
    )
    rotation_step = vector_norm(rotation_vector_world)
    if rotation_step > max_rotation_step_rad:
        return TwistEstimate(
            twist,
            False,
            f'rotation step {rotation_step:.3f} rad outside gate',
        )

    current_inverse = quaternion_conjugate(current_quaternion)
    linear_world = tuple(
        value / dt_sec for value in translation_delta_world
    )
    angular_world = tuple(
        value / dt_sec for value in rotation_vector_world
    )
    linear_child = rotate_vector(current_inverse, linear_world)
    angular_child = rotate_vector(current_inverse, angular_world)
    linear_speed = vector_norm(linear_child)
    angular_speed = vector_norm(angular_child)
    if linear_speed > max_linear_speed_mps:
        return TwistEstimate(
            twist,
            False,
            f'linear speed {linear_speed:.3f} m/s outside gate',
        )
    if angular_speed > max_angular_speed_radps:
        return TwistEstimate(
            twist,
            False,
            f'angular speed {angular_speed:.3f} rad/s outside gate',
        )

    twist.linear.x, twist.linear.y, twist.linear.z = linear_child
    twist.angular.x, twist.angular.y, twist.angular.z = angular_child
    return TwistEstimate(twist, True, 'valid')


def diagonal_covariance(
    linear_or_position_variance: float,
    angular_or_orientation_variance: float,
) -> list:
    """Create a row-major 6x6 covariance with two diagonal variances."""
    covariance = [0.0] * 36
    for index in (0, 7, 14):
        covariance[index] = linear_or_position_variance
    for index in (21, 28, 35):
        covariance[index] = angular_or_orientation_variance
    return covariance


def pose_to_transform(
    pose: PoseStamped,
    pelvis_frame: str,
) -> TransformStamped:
    """Convert a pelvis pose into its equivalent world-to-pelvis transform."""
    result = TransformStamped()
    result.header = pose.header
    result.child_frame_id = pelvis_frame
    result.transform.translation.x = pose.pose.position.x
    result.transform.translation.y = pose.pose.position.y
    result.transform.translation.z = pose.pose.position.z
    result.transform.rotation = pose.pose.orientation
    return result


class ApriltagLocalizationNode(Node):
    """Publish exact-time AprilTag-derived pelvis pose, TF, and odometry."""

    def __init__(self):
        """Create ROS interfaces and exact-time TF lookup."""
        super().__init__('apriltag_localization_node')
        self._declare_parameters()
        self._read_parameters()

        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self._queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        output_qos = QoSProfile(depth=self._queue_depth)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._pose_publisher = self.create_publisher(
            PoseStamped,
            self._pelvis_pose_topic,
            output_qos,
        )
        self._odom_publisher = self.create_publisher(
            Odometry,
            self._odom_topic,
            output_qos,
        )
        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray,
            self._diagnostics_topic,
            10,
        )
        self._pose_subscription = self.create_subscription(
            PoseStamped,
            self._tag_pose_topic,
            self._pose_callback,
            input_qos,
        )
        self._retry_timer = self.create_timer(
            1.0 / self._retry_rate_hz,
            self._drain_pending,
        )
        self._diagnostic_timer = self.create_timer(
            1.0 / self._diagnostic_rate_hz,
            self._publish_diagnostic,
        )

        self._pending: Deque[PendingPose] = deque()
        self._latest_received_stamp_nanoseconds: Optional[int] = None
        self._previous_pelvis_pose: Optional[PoseStamped] = None
        self._last_input_stamp_nanoseconds: Optional[int] = None
        self._last_output_stamp_nanoseconds: Optional[int] = None
        self._published_count = 0
        self._rejected_count = 0
        self._state = 'waiting_for_tag'
        self._state_level = DiagnosticStatus.WARN
        self._state_message = (
            f'waiting for {self._tag_pose_topic} with nonzero timestamps'
        )

        self.get_logger().info(
            f'Localizing {self._world_frame} -> {self._pelvis_frame} from '
            f'{self._tag_pose_topic} and exact-time '
            f'{self._tag_mount_frame} <- {self._pelvis_frame}; publishing '
            f'{self._pelvis_pose_topic} and {self._odom_topic}'
        )

    def _declare_parameters(self):
        defaults = {
            'tag_pose_topic': DEFAULT_TAG_POSE_TOPIC,
            'pelvis_pose_topic': DEFAULT_PELVIS_POSE_TOPIC,
            'odom_topic': DEFAULT_ODOM_TOPIC,
            'diagnostics_topic': DEFAULT_DIAGNOSTICS_TOPIC,
            'world_frame': DEFAULT_WORLD_FRAME,
            'detected_tag_frame': DEFAULT_DETECTED_TAG_FRAME,
            'tag_mount_frame': DEFAULT_TAG_MOUNT_FRAME,
            'pelvis_frame': DEFAULT_PELVIS_FRAME,
            'queue_depth': DEFAULT_QUEUE_DEPTH,
            'kinematic_wait_timeout_sec':
                DEFAULT_KINEMATIC_WAIT_TIMEOUT_SEC,
            'source_stale_timeout_sec':
                DEFAULT_SOURCE_STALE_TIMEOUT_SEC,
            'future_tolerance_sec': DEFAULT_FUTURE_TOLERANCE_SEC,
            'tf_lookup_timeout_sec': DEFAULT_TF_LOOKUP_TIMEOUT_SEC,
            'retry_rate_hz': DEFAULT_RETRY_RATE_HZ,
            'diagnostic_rate_hz': DEFAULT_DIAGNOSTIC_RATE_HZ,
            'pose_position_stddev_m':
                DEFAULT_POSE_POSITION_STDDEV_M,
            'pose_orientation_stddev_rad':
                DEFAULT_POSE_ORIENTATION_STDDEV_RAD,
            'twist_linear_stddev_mps':
                DEFAULT_TWIST_LINEAR_STDDEV_MPS,
            'twist_angular_stddev_radps':
                DEFAULT_TWIST_ANGULAR_STDDEV_RADPS,
            'twist_min_dt_sec': DEFAULT_TWIST_MIN_DT_SEC,
            'twist_max_dt_sec': DEFAULT_TWIST_MAX_DT_SEC,
            'twist_max_linear_speed_mps':
                DEFAULT_TWIST_MAX_LINEAR_SPEED_MPS,
            'twist_max_angular_speed_radps':
                DEFAULT_TWIST_MAX_ANGULAR_SPEED_RADPS,
            'twist_max_translation_step_m':
                DEFAULT_TWIST_MAX_TRANSLATION_STEP_M,
            'twist_max_rotation_step_rad':
                DEFAULT_TWIST_MAX_ROTATION_STEP_RAD,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        def get(name):
            return self.get_parameter(name).value

        (
            self._world_frame,
            self._detected_tag_frame,
            self._tag_mount_frame,
            self._pelvis_frame,
        ) = validate_frame_ids(
            get('world_frame'),
            get('detected_tag_frame'),
            get('tag_mount_frame'),
            get('pelvis_frame'),
        )
        self._tag_pose_topic = str(get('tag_pose_topic')).strip()
        self._pelvis_pose_topic = str(get('pelvis_pose_topic')).strip()
        self._odom_topic = str(get('odom_topic')).strip()
        self._diagnostics_topic = str(get('diagnostics_topic')).strip()
        for name in (
            '_tag_pose_topic',
            '_pelvis_pose_topic',
            '_odom_topic',
            '_diagnostics_topic',
        ):
            if not getattr(self, name):
                raise ValueError(f'{name[1:]} must not be empty')

        self._queue_depth = positive_integer(
            get('queue_depth'),
            'queue_depth',
        )
        self._kinematic_wait_timeout_sec = finite_positive(
            get('kinematic_wait_timeout_sec'),
            'kinematic_wait_timeout_sec',
        )
        self._source_stale_timeout_sec = finite_positive(
            get('source_stale_timeout_sec'),
            'source_stale_timeout_sec',
        )
        self._future_tolerance_sec = finite_nonnegative(
            get('future_tolerance_sec'),
            'future_tolerance_sec',
        )
        self._tf_lookup_timeout_sec = finite_nonnegative(
            get('tf_lookup_timeout_sec'),
            'tf_lookup_timeout_sec',
        )
        self._retry_rate_hz = finite_positive(
            get('retry_rate_hz'),
            'retry_rate_hz',
        )
        self._diagnostic_rate_hz = finite_positive(
            get('diagnostic_rate_hz'),
            'diagnostic_rate_hz',
        )
        for name in (
            'pose_position_stddev_m',
            'pose_orientation_stddev_rad',
            'twist_linear_stddev_mps',
            'twist_angular_stddev_radps',
            'twist_min_dt_sec',
            'twist_max_dt_sec',
            'twist_max_linear_speed_mps',
            'twist_max_angular_speed_radps',
            'twist_max_translation_step_m',
            'twist_max_rotation_step_rad',
        ):
            setattr(self, f'_{name}', finite_positive(get(name), name))
        if self._twist_min_dt_sec > self._twist_max_dt_sec:
            raise ValueError(
                'twist_min_dt_sec must not exceed twist_max_dt_sec'
            )

        self._pose_covariance = diagonal_covariance(
            self._pose_position_stddev_m ** 2,
            self._pose_orientation_stddev_rad ** 2,
        )
        self._valid_twist_covariance = diagonal_covariance(
            self._twist_linear_stddev_mps ** 2,
            self._twist_angular_stddev_radps ** 2,
        )
        self._unknown_twist_covariance = diagonal_covariance(
            UNKNOWN_VARIANCE,
            UNKNOWN_VARIANCE,
        )
        self._tf_lookup_timeout = Duration(
            seconds=self._tf_lookup_timeout_sec
        )

    def _pose_callback(self, message: PoseStamped):
        now_nanoseconds = self.get_clock().now().nanoseconds
        source_stamp = stamp_nanoseconds(message.header.stamp)
        if source_stamp is None:
            self._reject('tag pose has a zero timestamp', state='stale')
            return
        age_sec = (now_nanoseconds - source_stamp) * 1.0e-9
        if age_sec < -self._future_tolerance_sec:
            self._reject(
                f'tag pose is {-age_sec:.3f} s in the future',
                state='stale',
            )
            return
        if age_sec > self._source_stale_timeout_sec:
            self._reject(
                f'tag pose is stale by {age_sec:.3f} s',
                state='stale',
            )
            return
        if (
            self._latest_received_stamp_nanoseconds is not None
            and source_stamp <= self._latest_received_stamp_nanoseconds
        ):
            self._reject(
                'tag pose timestamp is duplicate or out of order',
                state=self._state,
            )
            return
        try:
            pose = normalized_pose(message, self._world_frame)
        except ValueError as exc:
            self._reject(str(exc), state='stale')
            return

        self._latest_received_stamp_nanoseconds = source_stamp
        self._last_input_stamp_nanoseconds = source_stamp
        if len(self._pending) >= self._queue_depth:
            self._pending.popleft()
            self._rejected_count += 1
            self._warn('kinematic wait queue full; dropped oldest tag pose')
        self._pending.append(
            PendingPose(pose, source_stamp, now_nanoseconds)
        )
        self._drain_pending()

    def _drain_pending(self):
        now_nanoseconds = self.get_clock().now().nanoseconds
        while self._pending:
            pending = self._pending[0]
            source_age = (
                now_nanoseconds - pending.stamp_nanoseconds
            ) * 1.0e-9
            wait_age = (
                now_nanoseconds - pending.enqueue_nanoseconds
            ) * 1.0e-9
            if source_age > self._source_stale_timeout_sec:
                self._pending.popleft()
                self._reject(
                    f'tag pose became stale after {source_age:.3f} s',
                    state='stale',
                )
                continue

            request_time = Time(
                nanoseconds=pending.stamp_nanoseconds,
                clock_type=self.get_clock().clock_type,
            )
            try:
                kinematics = self._tf_buffer.lookup_transform(
                    self._tag_mount_frame,
                    self._pelvis_frame,
                    request_time,
                    timeout=self._tf_lookup_timeout,
                )
            except TransformException as exc:
                if wait_age >= self._kinematic_wait_timeout_sec:
                    self._pending.popleft()
                    self._reject(
                        'exact-time robot kinematics unavailable after '
                        f'{wait_age:.3f} s: {exc}',
                        state='stale',
                    )
                    continue
                self._set_state(
                    'waiting_for_kinematics',
                    DiagnosticStatus.WARN,
                    f'waiting for {self._tag_mount_frame} <- '
                    f'{self._pelvis_frame} at tag stamp',
                )
                break

            try:
                pelvis_pose = compose_pelvis_pose(
                    pending.pose,
                    kinematics,
                    world_frame=self._world_frame,
                    tag_mount_frame=self._tag_mount_frame,
                    pelvis_frame=self._pelvis_frame,
                )
            except ValueError as exc:
                self._pending.popleft()
                self._reject(str(exc), state='stale')
                continue

            self._pending.popleft()
            self._publish_localization(pelvis_pose)

    def _publish_localization(self, pelvis_pose: PoseStamped):
        twist_estimate = estimate_child_twist(
            self._previous_pelvis_pose,
            pelvis_pose,
            min_dt_sec=self._twist_min_dt_sec,
            max_dt_sec=self._twist_max_dt_sec,
            max_linear_speed_mps=self._twist_max_linear_speed_mps,
            max_angular_speed_radps=self._twist_max_angular_speed_radps,
            max_translation_step_m=self._twist_max_translation_step_m,
            max_rotation_step_rad=self._twist_max_rotation_step_rad,
        )
        # Always advance the finite-difference baseline.  A failed derivative
        # gate invalidates only this twist, never a valid localization pose.
        self._previous_pelvis_pose = pelvis_pose

        odometry = Odometry()
        odometry.header = pelvis_pose.header
        odometry.child_frame_id = self._pelvis_frame
        odometry.pose.pose = pelvis_pose.pose
        odometry.pose.covariance = self._pose_covariance
        odometry.twist.twist = twist_estimate.twist
        odometry.twist.covariance = (
            self._valid_twist_covariance
            if twist_estimate.valid
            else self._unknown_twist_covariance
        )

        self._tf_broadcaster.sendTransform(
            pose_to_transform(pelvis_pose, self._pelvis_frame)
        )
        self._pose_publisher.publish(pelvis_pose)
        self._odom_publisher.publish(odometry)
        self._last_output_stamp_nanoseconds = stamp_nanoseconds(
            pelvis_pose.header.stamp
        )
        self._published_count += 1
        self._set_state(
            'active',
            DiagnosticStatus.OK,
            'publishing exact-time AprilTag pelvis localization'
            + (
                ''
                if twist_estimate.valid
                else f'; twist unavailable: {twist_estimate.reason}'
            ),
        )

    def _reject(self, message: str, *, state: str):
        self._rejected_count += 1
        level = (
            DiagnosticStatus.WARN
            if state in ('waiting_for_tag', 'waiting_for_kinematics')
            else DiagnosticStatus.ERROR
        )
        self._set_state(state, level, message)
        self._warn(message)

    def _set_state(self, state: str, level: int, message: str):
        self._state = state
        self._state_level = level
        self._state_message = message

    def _publish_diagnostic(self):
        now = self.get_clock().now()
        now_nanoseconds = now.nanoseconds
        if (
            self._state == 'active'
            and self._last_output_stamp_nanoseconds is not None
            and (
                now_nanoseconds - self._last_output_stamp_nanoseconds
            ) * 1.0e-9 > self._source_stale_timeout_sec
        ):
            self._set_state(
                'stale',
                DiagnosticStatus.ERROR,
                'latest AprilTag pelvis localization is stale',
            )

        message = DiagnosticArray()
        message.header.stamp = now.to_msg()
        status = DiagnosticStatus()
        status.name = 'g1_navigation/apriltag_localization'
        status.hardware_id = 'g1'
        status.level = self._state_level
        status.message = self._state_message
        status.values = [
            KeyValue(key='state', value=self._state),
            KeyValue(key='tag_pose_topic', value=self._tag_pose_topic),
            KeyValue(
                key='kinematic_lookup',
                value=(
                    f'{self._tag_mount_frame} <- {self._pelvis_frame}'
                ),
            ),
            KeyValue(
                key='output_tf',
                value=f'{self._world_frame} -> {self._pelvis_frame}',
            ),
            KeyValue(key='queue_size', value=str(len(self._pending))),
            KeyValue(
                key='published_count',
                value=str(self._published_count),
            ),
            KeyValue(
                key='rejected_count',
                value=str(self._rejected_count),
            ),
        ]
        message.status = [status]
        self._diagnostic_publisher.publish(message)

    def _warn(self, message: str):
        self.get_logger().warn(
            message,
            throttle_duration_sec=WARNING_THROTTLE_SEC,
        )


def main(args=None):
    """Run the AprilTag localization node until ROS shuts down."""
    rclpy.init(args=args)
    node = ApriltagLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

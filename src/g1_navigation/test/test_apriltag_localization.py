from collections import deque
import math

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped, TransformStamped
import pytest
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import TransformException

import g1_navigation.apriltag_localization as localization_module
from g1_navigation.apriltag_localization import ApriltagLocalizationNode
from g1_navigation.apriltag_localization import compose_pelvis_pose
from g1_navigation.apriltag_localization import diagonal_covariance
from g1_navigation.apriltag_localization import estimate_child_twist
from g1_navigation.apriltag_localization import normalize_quaternion
from g1_navigation.apriltag_localization import normalized_pose
from g1_navigation.apriltag_localization import pose_to_transform
from g1_navigation.apriltag_localization import stamp_nanoseconds
from g1_navigation.apriltag_localization import validate_frame_ids
from g1_navigation.apriltag_localization import UNKNOWN_VARIANCE
from g1_navigation.apriltag_localization import WARNING_THROTTLE_SEC


def _pose(
    *,
    stamp=(12, 0),
    frame='world',
    translation=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    result = PoseStamped()
    result.header.stamp.sec, result.header.stamp.nanosec = stamp
    result.header.frame_id = frame
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


def _kinematics(
    *,
    stamp=(12, 0),
    parent='tag_mount_frame',
    child='pelvis',
    translation=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
):
    result = TransformStamped()
    result.header.stamp.sec, result.header.stamp.nanosec = stamp
    result.header.frame_id = parent
    result.child_frame_id = child
    (
        result.transform.translation.x,
        result.transform.translation.y,
        result.transform.translation.z,
    ) = translation
    (
        result.transform.rotation.x,
        result.transform.rotation.y,
        result.transform.rotation.z,
        result.transform.rotation.w,
    ) = quaternion
    return result


class _Clock:

    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds
        self.clock_type = Time(nanoseconds=nanoseconds).clock_type

    def now(self):
        return Time(
            nanoseconds=self.nanoseconds,
            clock_type=self.clock_type,
        )


class _Buffer:

    def __init__(self, results):
        self.results = deque(results)
        self.calls = []

    def lookup_transform(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


class _Publisher:

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class _Broadcaster:

    def __init__(self):
        self.transforms = []

    def sendTransform(self, transform):
        self.transforms.append(transform)


class _Logger:

    def __init__(self):
        self.calls = []

    def warn(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _NodeHarness:

    _pose_callback = ApriltagLocalizationNode._pose_callback
    _drain_pending = ApriltagLocalizationNode._drain_pending
    _publish_localization = ApriltagLocalizationNode._publish_localization
    _publish_diagnostic = ApriltagLocalizationNode._publish_diagnostic
    _reject = ApriltagLocalizationNode._reject
    _set_state = ApriltagLocalizationNode._set_state

    def __init__(self, results, now_nanoseconds=12_100_000_000):
        self._world_frame = 'world'
        self._detected_tag_frame = 'tag_frame'
        self._tag_mount_frame = 'tag_mount_frame'
        self._pelvis_frame = 'pelvis'
        self._tag_pose_topic = '/pose/tag_frame'
        self._queue_depth = 20
        self._kinematic_wait_timeout_sec = 0.2
        self._source_stale_timeout_sec = 0.5
        self._future_tolerance_sec = 0.05
        self._tf_lookup_timeout = Duration(seconds=0.0)
        self._twist_min_dt_sec = 0.02
        self._twist_max_dt_sec = 0.5
        self._twist_max_linear_speed_mps = 3.0
        self._twist_max_angular_speed_radps = 4.0
        self._twist_max_translation_step_m = 0.5
        self._twist_max_rotation_step_rad = math.radians(45.0)
        self._pose_covariance = diagonal_covariance(
            0.05**2,
            math.radians(5)**2,
        )
        self._valid_twist_covariance = diagonal_covariance(1.0, 1.0)
        self._unknown_twist_covariance = diagonal_covariance(
            UNKNOWN_VARIANCE,
            UNKNOWN_VARIANCE,
        )
        self._pending = deque()
        self._latest_received_stamp_nanoseconds = None
        self._previous_pelvis_pose = None
        self._last_input_stamp_nanoseconds = None
        self._last_output_stamp_nanoseconds = None
        self._published_count = 0
        self._rejected_count = 0
        self._state = 'waiting_for_tag'
        self._state_level = DiagnosticStatus.WARN
        self._state_message = 'waiting'
        self._tf_buffer = _Buffer(results)
        self._tf_broadcaster = _Broadcaster()
        self._pose_publisher = _Publisher()
        self._odom_publisher = _Publisher()
        self._diagnostic_publisher = _Publisher()
        self._clock = _Clock(now_nanoseconds)
        self.warnings = []

    def get_clock(self):
        return self._clock

    def _warn(self, message):
        self.warnings.append(message)


def _xyz(message):
    return (message.x, message.y, message.z)


def _xyzw(message):
    return (message.x, message.y, message.z, message.w)


def test_noncommuting_se3_composition_preserves_all_axes():
    root_half = math.sqrt(0.5)
    tag = _pose(
        translation=(1.0, 2.0, 3.0),
        quaternion=(0.0, 0.0, root_half, root_half),
    )
    mount = _kinematics(
        translation=(1.0, 0.0, 0.0),
        quaternion=(root_half, 0.0, 0.0, root_half),
    )

    result = compose_pelvis_pose(
        tag,
        mount,
        world_frame='world',
        tag_mount_frame='tag_mount_frame',
        pelvis_frame='pelvis',
    )

    assert _xyz(result.pose.position) == pytest.approx((1.0, 3.0, 3.0))
    # Rz(90) Rx(90), not an Euler-component sum.
    assert _xyzw(result.pose.orientation) == pytest.approx(
        (0.5, 0.5, 0.5, 0.5)
    )
    assert result.header.stamp == tag.header.stamp
    assert result.header.frame_id == 'world'


def test_pose_and_kinematics_are_normalized_and_validated():
    tag = normalized_pose(
        _pose(quaternion=(0.0, 0.0, 0.0, -2.0)),
        'world',
    )
    mount = _kinematics(quaternion=(0.0, 0.0, 3.0, 3.0))

    result = compose_pelvis_pose(
        tag,
        mount,
        world_frame='world',
        tag_mount_frame='tag_mount_frame',
        pelvis_frame='pelvis',
    )

    assert math.sqrt(sum(v * v for v in _xyzw(
        result.pose.orientation
    ))) == pytest.approx(1.0)
    with pytest.raises(ValueError, match='expected world'):
        normalized_pose(_pose(frame='camera'), 'world')
    with pytest.raises(ValueError, match='unexpected kinematic'):
        compose_pelvis_pose(
            tag,
            _kinematics(parent='wrong'),
            world_frame='world',
            tag_mount_frame='tag_mount_frame',
            pelvis_frame='pelvis',
        )


@pytest.mark.parametrize(
    'quaternion',
    [
        (math.nan, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_invalid_quaternions_fail_closed(quaternion):
    with pytest.raises(ValueError):
        normalize_quaternion(quaternion)


def test_twist_is_reported_in_current_pelvis_frame():
    root_half = math.sqrt(0.5)
    previous = _pose(
        stamp=(12, 0),
        quaternion=(0.0, 0.0, root_half, root_half),
    )
    current = _pose(
        stamp=(12, 100_000_000),
        translation=(0.0, 0.1, 0.0),
        quaternion=(0.0, 0.0, root_half, root_half),
    )

    estimate = estimate_child_twist(
        previous,
        current,
        min_dt_sec=0.02,
        max_dt_sec=0.5,
        max_linear_speed_mps=3.0,
        max_angular_speed_radps=4.0,
        max_translation_step_m=0.5,
        max_rotation_step_rad=math.radians(45),
    )

    assert estimate.valid
    assert _xyz(estimate.twist.linear) == pytest.approx((1.0, 0.0, 0.0))
    assert _xyz(estimate.twist.angular) == pytest.approx((0.0, 0.0, 0.0))


def test_twist_handles_quaternion_sign_and_three_axis_rotation():
    angle = 0.1
    axis = (1.0, 2.0, 3.0)
    axis_norm = math.sqrt(sum(value * value for value in axis))
    scale = math.sin(angle / 2.0) / axis_norm
    current_quaternion = tuple(value * scale for value in axis) + (
        math.cos(angle / 2.0),
    )
    previous = _pose(stamp=(12, 0))
    current = _pose(
        stamp=(12, 100_000_000),
        quaternion=tuple(-value for value in current_quaternion),
    )

    estimate = estimate_child_twist(
        previous,
        current,
        min_dt_sec=0.02,
        max_dt_sec=0.5,
        max_linear_speed_mps=3.0,
        max_angular_speed_radps=4.0,
        max_translation_step_m=0.5,
        max_rotation_step_rad=math.radians(45),
    )

    assert estimate.valid
    assert math.sqrt(sum(
        value * value for value in _xyz(estimate.twist.angular)
    )) == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in _xyz(
        estimate.twist.angular
    ))


@pytest.mark.parametrize(
    ('stamp', 'translation', 'quaternion', 'reason'),
    [
        ((12, 10_000_000), (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0, 1.0), 'dt'),
        ((13, 0), (0.0, 0.0, 0.0),
         (0.0, 0.0, 0.0, 1.0), 'dt'),
        ((12, 100_000_000), (0.6, 0.0, 0.0),
         (0.0, 0.0, 0.0, 1.0), 'translation step'),
        ((12, 100_000_000), (0.4, 0.0, 0.0),
         (0.0, 0.0, 0.0, 1.0), 'linear speed'),
        ((12, 100_000_000), (0.0, 0.0, 0.0),
         (0.0, 0.0, math.sin(0.5), math.cos(0.5)), 'rotation step'),
    ],
)
def test_twist_gates_return_zero_unknown_twist(
    stamp,
    translation,
    quaternion,
    reason,
):
    estimate = estimate_child_twist(
        _pose(stamp=(12, 0)),
        _pose(
            stamp=stamp,
            translation=translation,
            quaternion=quaternion,
        ),
        min_dt_sec=0.02,
        max_dt_sec=0.5,
        max_linear_speed_mps=3.0,
        max_angular_speed_radps=4.0,
        max_translation_step_m=0.5,
        max_rotation_step_rad=math.radians(45),
    )

    assert not estimate.valid
    assert reason in estimate.reason
    assert _xyz(estimate.twist.linear) == (0.0, 0.0, 0.0)
    assert _xyz(estimate.twist.angular) == (0.0, 0.0, 0.0)


def test_event_callback_looks_up_exact_stamp_and_publishes_matching_outputs():
    node = _NodeHarness([
        _kinematics(
            translation=(0.25, -0.5, 0.75),
            quaternion=(0.0, 0.0, 0.0, 2.0),
        )
    ])
    tag = _pose(
        translation=(1.0, 2.0, 3.0),
        quaternion=(0.0, 0.0, 0.0, 2.0),
    )

    node._pose_callback(tag)

    assert not node._pending
    assert len(node._tf_buffer.calls) == 1
    args, kwargs = node._tf_buffer.calls[0]
    assert args[:2] == ('tag_mount_frame', 'pelvis')
    assert args[2].nanoseconds == 12_000_000_000
    assert kwargs['timeout'].nanoseconds == 0
    assert len(node._tf_broadcaster.transforms) == 1
    assert len(node._pose_publisher.messages) == 1
    assert len(node._odom_publisher.messages) == 1

    transform = node._tf_broadcaster.transforms[0]
    pose = node._pose_publisher.messages[0]
    odom = node._odom_publisher.messages[0]
    assert transform.header.stamp == tag.header.stamp
    assert pose.header.stamp == tag.header.stamp
    assert odom.header.stamp == tag.header.stamp
    assert transform.header.frame_id == 'world'
    assert transform.child_frame_id == 'pelvis'
    assert odom.header.frame_id == 'world'
    assert odom.child_frame_id == 'pelvis'
    assert _xyz(pose.pose.position) == pytest.approx((1.25, 1.5, 3.75))
    assert odom.twist.covariance[0] == UNKNOWN_VARIANCE
    assert node._state == 'active'


def test_delayed_kinematics_are_retried_at_the_original_timestamp():
    node = _NodeHarness([
        TransformException('future extrapolation'),
        _kinematics(),
    ])
    node._pose_callback(_pose())

    assert len(node._pending) == 1
    assert not node._pose_publisher.messages
    assert node._state == 'waiting_for_kinematics'

    node._clock.nanoseconds += 100_000_000
    node._drain_pending()

    assert not node._pending
    assert len(node._pose_publisher.messages) == 1
    lookup_times = [call[0][2].nanoseconds for call in node._tf_buffer.calls]
    assert lookup_times == [12_000_000_000, 12_000_000_000]


def test_missing_joint_history_expires_fail_closed():
    node = _NodeHarness([
        TransformException('missing waist joint TF'),
        TransformException('still missing'),
    ])
    node._pose_callback(_pose())
    node._clock.nanoseconds += 210_000_000
    node._drain_pending()

    assert not node._pending
    assert not node._tf_broadcaster.transforms
    assert not node._pose_publisher.messages
    assert not node._odom_publisher.messages
    assert node._state == 'stale'
    assert any('exact-time robot kinematics unavailable' in message
               for message in node.warnings)


@pytest.mark.parametrize(
    ('message', 'warning'),
    [
        (_pose(stamp=(0, 0)), 'zero timestamp'),
        (_pose(stamp=(11, 0)), 'stale'),
        (_pose(stamp=(13, 0)), 'future'),
        (_pose(frame='camera'), 'expected world'),
    ],
)
def test_invalid_inputs_are_rejected_without_tf_or_odometry(message, warning):
    node = _NodeHarness([])

    node._pose_callback(message)

    assert not node._pending
    assert not node._tf_broadcaster.transforms
    assert not node._odom_publisher.messages
    assert any(warning in value for value in node.warnings)


def test_duplicate_and_out_of_order_inputs_are_rejected():
    node = _NodeHarness([_kinematics()])
    node._pose_callback(_pose(stamp=(12, 0)))
    node._pose_callback(_pose(stamp=(12, 0)))
    node._pose_callback(_pose(stamp=(11, 999_999_999)))

    assert len(node._pose_publisher.messages) == 1
    assert node._rejected_count == 2
    assert sum(
        'duplicate or out of order' in warning for warning in node.warnings
    ) == 2


def test_bad_twist_does_not_suppress_or_modify_pose():
    node = _NodeHarness([_kinematics(), _kinematics(stamp=(12, 100_000_000))])
    node._pose_callback(_pose(stamp=(12, 0)))
    node._clock.nanoseconds = 12_200_000_000
    node._pose_callback(
        _pose(stamp=(12, 100_000_000), translation=(0.6, 0.0, 0.0))
    )

    assert len(node._pose_publisher.messages) == 2
    assert _xyz(
        node._pose_publisher.messages[-1].pose.position
    ) == pytest.approx((0.6, 0.0, 0.0))
    assert node._odom_publisher.messages[-1].twist.covariance[0] == (
        UNKNOWN_VARIANCE
    )
    assert node._previous_pelvis_pose.header.stamp.sec == 12
    assert node._previous_pelvis_pose.header.stamp.nanosec == 100_000_000


def test_valid_twist_covariance_and_pose_covariance_are_published():
    node = _NodeHarness([_kinematics(), _kinematics(stamp=(12, 100_000_000))])
    node._pose_callback(_pose(stamp=(12, 0)))
    node._clock.nanoseconds = 12_200_000_000
    node._pose_callback(
        _pose(stamp=(12, 100_000_000), translation=(0.1, 0.0, 0.0))
    )

    odom = node._odom_publisher.messages[-1]
    assert odom.pose.covariance[0] == pytest.approx(0.05**2)
    assert odom.pose.covariance[21] == pytest.approx(math.radians(5)**2)
    assert odom.twist.covariance[0] == pytest.approx(1.0)
    assert odom.twist.covariance[21] == pytest.approx(1.0)
    assert odom.twist.twist.linear.x == pytest.approx(1.0)


def test_active_diagnostic_becomes_stale_after_input_timeout():
    node = _NodeHarness([_kinematics()])
    node._pose_callback(_pose())
    node._publish_diagnostic()

    active = node._diagnostic_publisher.messages[-1].status[0]
    assert active.level == DiagnosticStatus.OK
    assert any(
        value.key == 'state' and value.value == 'active'
        for value in active.values
    )

    node._clock.nanoseconds = 12_600_000_000
    node._publish_diagnostic()
    stale = node._diagnostic_publisher.messages[-1].status[0]
    assert stale.level == DiagnosticStatus.ERROR
    assert stale.values[0].value == 'stale'


def test_pose_to_transform_keeps_pose_stamp_and_world_parent():
    pose = _pose(translation=(1.0, 2.0, 3.0))

    transform = pose_to_transform(pose, 'pelvis')

    assert transform.header == pose.header
    assert transform.child_frame_id == 'pelvis'
    assert _xyz(transform.transform.translation) == (1.0, 2.0, 3.0)


def test_frame_ids_are_normalized_and_distinct():
    assert validate_frame_ids(
        ' /world ',
        '/tag_frame',
        ' tag_mount_frame ',
        'pelvis',
    ) == ('world', 'tag_frame', 'tag_mount_frame', 'pelvis')
    with pytest.raises(ValueError, match='must not be empty'):
        validate_frame_ids('', 'tag_frame', 'tag_mount_frame', 'pelvis')
    with pytest.raises(ValueError, match='must be distinct'):
        validate_frame_ids(
            'world',
            '/world',
            'tag_mount_frame',
            'pelvis',
        )


def test_runtime_warnings_are_throttled():
    logger = _Logger()
    harness = type(
        'LoggerHarness',
        (),
        {'get_logger': lambda self: logger},
    )()

    ApriltagLocalizationNode._warn(harness, 'retrying')

    assert logger.calls == [
        (
            ('retrying',),
            {'throttle_duration_sec': WARNING_THROTTLE_SEC},
        )
    ]


def test_stamp_conversion_rejects_zero():
    assert stamp_nanoseconds(_pose(stamp=(0, 0)).header.stamp) is None
    assert stamp_nanoseconds(_pose(stamp=(7, 9)).header.stamp) == (
        7_000_000_009
    )


def test_main_handles_launch_interrupt_after_context_shutdown(monkeypatch):
    calls = []

    class _MainNode:

        def destroy_node(self):
            calls.append('destroy')

    node = _MainNode()
    monkeypatch.setattr(
        localization_module.rclpy,
        'init',
        lambda args=None: calls.append(('init', args)),
    )
    monkeypatch.setattr(
        localization_module,
        'ApriltagLocalizationNode',
        lambda: node,
    )

    def interrupt(_node):
        calls.append('spin')
        raise KeyboardInterrupt

    monkeypatch.setattr(localization_module.rclpy, 'spin', interrupt)
    monkeypatch.setattr(localization_module.rclpy, 'ok', lambda: False)
    monkeypatch.setattr(
        localization_module.rclpy,
        'shutdown',
        lambda: calls.append('shutdown'),
    )

    localization_module.main(args=['--test'])

    assert calls == [('init', ['--test']), 'spin', 'destroy']

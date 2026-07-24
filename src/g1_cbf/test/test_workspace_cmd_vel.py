import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from g1_cbf_msg.msg import WorkspaceState
from sensor_msgs.msg import Joy
from std_msgs.msg import String


_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'workspace_cmd_vel_node.py'
_SPEC = importlib.util.spec_from_file_location(
    'workspace_cmd_vel_node_script', _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_interpolated_path_is_two_by_five_metres_and_respects_resolution():
    samples = _MODULE.interpolate_path(0.05)
    cursor = 0
    for x, y in _MODULE.ROUTE_POINTS:
        while cursor < len(samples):
            sx, sy = samples[cursor]
            if np.linalg.norm(np.array([sx - x, sy - y])) < 1e-9:
                break
            cursor += 1
        assert cursor < len(samples)
        cursor += 1

    for previous, current in zip(samples[:-1], samples[1:]):
        distance = np.linalg.norm(np.array(current) - np.array(previous))
        assert distance <= 0.050001
    xy = np.asarray(samples)
    assert np.ptp(xy[:, 0]) == pytest.approx(2.0)
    assert np.ptp(xy[:, 1]) == pytest.approx(5.0)
    assert samples[-1] == pytest.approx([0.0, 0.0])


def test_target_advances_at_point_two_five_with_fixed_robot():
    follower = _MODULE.RectangleFollower()
    position = np.zeros(2)

    velocity, yaw_rate = follower.command(position, 0.0, elapsed_sec=1.0)
    assert follower.target_distance == pytest.approx(0.25)
    assert follower.target == pytest.approx([0.25, 0.0])
    assert np.linalg.norm(velocity) == pytest.approx(0.25)
    assert yaw_rate == pytest.approx(0.0)

    follower.command(position, 0.0, elapsed_sec=1.0)
    assert follower.target_distance == pytest.approx(0.50)
    assert follower.target == pytest.approx([0.50, 0.0])


def test_initial_and_slightly_ahead_robot_use_forward_path_tangent():
    follower = _MODULE.RectangleFollower()

    velocity, yaw_rate = follower.command(np.zeros(2), 0.0)
    assert velocity == pytest.approx([0.25, 0.0])
    assert yaw_rate == 0.0

    velocity, yaw_rate = follower.command(np.array([0.05, 0.0]), 0.0)
    assert velocity == pytest.approx([0.25, 0.0])
    assert yaw_rate == 0.0


def test_target_progress_is_not_gated_by_robot_or_workspace_circle():
    follower = _MODULE.RectangleFollower()
    blocked_position = np.array([0.0, 1.4])

    follower.command(blocked_position, 0.0, elapsed_sec=10.0)
    assert follower.target == pytest.approx([1.0, 1.5])
    follower.command(blocked_position, 0.0, elapsed_sec=10.0)
    assert follower.target == pytest.approx([-0.5, 2.5])


def test_velocity_and_heading_always_point_at_current_target():
    follower = _MODULE.RectangleFollower(heading_kp=1.5)
    position = np.array([0.0, 0.0])
    velocity, yaw_rate = follower.command(
        position,
        0.0,
        elapsed_sec=5.0,
    )

    target_direction = follower.target - position
    target_direction /= np.linalg.norm(target_direction)
    assert velocity / np.linalg.norm(velocity) == pytest.approx(
        target_direction
    )
    assert np.linalg.norm(velocity) == pytest.approx(0.25)
    expected_bearing = math.atan2(target_direction[1], target_direction[0])
    assert yaw_rate == pytest.approx(1.5 * expected_bearing)


def test_elapsed_zero_pauses_and_reset_returns_target_to_origin():
    follower = _MODULE.RectangleFollower()
    follower.command(np.zeros(2), 0.0, elapsed_sec=2.0)
    paused_target = follower.target.copy()

    follower.command(np.zeros(2), 0.0, elapsed_sec=0.0)
    assert follower.target == pytest.approx(paused_target)
    follower.reset()
    assert follower.target_distance == 0.0
    assert follower.target == pytest.approx([0.0, 0.0])


class _FakeTime:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def __sub__(self, other):
        return _FakeTime(self.nanoseconds - other.nanoseconds)


def test_path_clock_resumes_without_counting_paused_time():
    times = iter((
        _FakeTime(1_000_000_000),
        _FakeTime(1_100_000_000),
        _FakeTime(5_000_000_000),
    ))
    node = SimpleNamespace(
        _last_path_tick_time=None,
        get_clock=lambda: SimpleNamespace(now=lambda: next(times)),
    )

    assert _MODULE.WorkspaceCmdVelNode._path_elapsed_sec(node) == 0.0
    elapsed = _MODULE.WorkspaceCmdVelNode._path_elapsed_sec(node)
    assert elapsed == pytest.approx(0.1)
    _MODULE.WorkspaceCmdVelNode._pause_path_progress(node)
    assert _MODULE.WorkspaceCmdVelNode._path_elapsed_sec(node) == 0.0


def test_follower_waits_at_final_target_until_robot_arrives():
    follower = _MODULE.RectangleFollower()
    far_position = np.array([-1.0, 0.0])
    velocity, _ = follower.command(
        far_position,
        0.0,
        elapsed_sec=follower.total_length / follower.speed + 1.0,
    )
    assert follower.target == pytest.approx([0.0, 0.0])
    assert not follower.finished
    assert velocity == pytest.approx([0.25, 0.0])

    velocity, yaw_rate = follower.command(
        np.array([-0.05, 0.0]),
        0.0,
        elapsed_sec=0.0,
    )
    assert follower.finished
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def test_path_poses_use_identity_orientation():
    publisher = _Publisher()
    node = SimpleNamespace(
        _workspace_frame='workspace',
        _path_resolution=0.05,
        _path_pub=publisher,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=8))
        ),
    )
    _MODULE.WorkspaceCmdVelNode._publish_path(node)

    msg = publisher.messages[0]
    assert msg.header.frame_id == 'workspace'
    assert msg.header.stamp.sec == 8
    assert all(pose.pose.orientation.w == 1.0 for pose in msg.poses)
    assert all(pose.pose.orientation.x == 0.0 for pose in msg.poses)
    assert all(pose.pose.orientation.y == 0.0 for pose in msg.poses)
    assert all(pose.pose.orientation.z == 0.0 for pose in msg.poses)


def test_planar_reference_point_is_exact_active_target_in_workspace():
    publisher = _Publisher()
    node = SimpleNamespace(
        _workspace_frame='workspace',
        _planar_reference_pub=publisher,
        get_clock=lambda: SimpleNamespace(
            now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=9))
        ),
    )
    target = np.array([1.0, 0.4])
    _MODULE.WorkspaceCmdVelNode._publish_planar_reference(node, target)

    msg = publisher.messages[0]
    assert msg.header.frame_id == 'workspace'
    assert msg.header.stamp.sec == 9
    assert [msg.point.x, msg.point.y, msg.point.z] == pytest.approx(
        [1.0, 0.4, 0.0]
    )


def test_workspace_to_body_supports_holonomic_motion():
    body = _MODULE.workspace_to_body(np.array([0.5, 0.2]), math.pi / 2)
    assert body == pytest.approx([0.2, -0.5])


class _Follower:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def _authority_node(workspace_enabled=True, orchestrator_required=True,
                    workspace_cbf_available=True):
    follower = _Follower()
    node = SimpleNamespace(
        _authority=_MODULE.AUTHORITY_JOYSTICK,
        _orchestrator_required=orchestrator_required,
        _orchestrator_state='control',
        _buttons_armed=True,
        _last_center_button=False,
        _last_path_button=False,
        _workspace_enabled=workspace_enabled,
        _workspace_cbf_available=workspace_cbf_available,
        _capture_pending=False,
        _capture_pause=False,
        _workspace_generation=0,
        _await_tf_stamp_ns=None,
        _follower=follower,
        _pause_path_progress=lambda: None,
        _control_available=lambda: True,
        _button_pressed=_MODULE.WorkspaceCmdVelNode._button_pressed,
        _reset_center_history=lambda: None,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        _publish_path=lambda: None,
    )

    def set_authority(authority):
        node._authority = authority

    node._set_authority = set_authority
    return node


def _joy(*buttons, axes=None):
    msg = Joy()
    msg.axes = list(axes) if axes is not None else [0.0] * 8
    msg.buttons = [0] * 11
    for button in buttons:
        msg.buttons[button] = 1
    return msg


def test_button_toggles_and_center_precedence():
    node = _authority_node()
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_PATH
    # Holding the button produces no additional toggle.
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_PATH
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    assert node._authority == _MODULE.AUTHORITY_PATH

    # Center wins when both rising edges arrive in one sample.
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(0, 1))
    assert node._authority == _MODULE.AUTHORITY_CENTER
    assert node._follower.reset_count == 1

    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_CENTER
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(0))
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK


def test_path_toggle_pauses_and_resumes_without_resetting_cursor():
    node = _authority_node()
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_PATH
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_PATH
    assert node._follower.reset_count == 0


def test_center_button_is_ignored_when_workspace_disabled():
    node = _authority_node(workspace_enabled=False)
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(0, 1))
    assert node._authority == _MODULE.AUTHORITY_PATH
    assert node._follower.reset_count == 0


def test_center_button_is_ignored_when_workspace_cbf_is_unavailable():
    node = _authority_node(workspace_cbf_available=False)
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(0))
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK
    assert node._follower.reset_count == 0


def test_button_three_does_not_locally_infer_workspace_pending_state():
    node = _authority_node()
    node._authority = _MODULE.AUTHORITY_PATH
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(3))
    assert not node._capture_pause
    assert node._authority == _MODULE.AUTHORITY_PATH


def test_joystick_axes_never_change_authority():
    node = _authority_node()
    _MODULE.WorkspaceCmdVelNode._joy_cb(
        node, _joy(axes=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    )
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK


def test_workspace_pending_and_generation_reset_behavior():
    node = _authority_node()
    node._authority = _MODULE.AUTHORITY_CENTER

    pending = WorkspaceState()
    pending.generation = 0
    pending.enabled = False
    pending.capture_pending = True
    _MODULE.WorkspaceCmdVelNode._workspace_state_cb(node, pending)
    assert node._authority == _MODULE.AUTHORITY_CENTER
    assert node._capture_pause

    captured = WorkspaceState()
    captured.header.stamp.sec = 12
    captured.generation = 1
    captured.enabled = True
    captured.capture_pending = False
    _MODULE.WorkspaceCmdVelNode._workspace_state_cb(node, captured)
    assert node._authority == _MODULE.AUTHORITY_CENTER
    assert not node._capture_pause
    assert node._follower.reset_count == 1
    assert node._await_tf_stamp_ns == 12_000_000_000

    disabled = WorkspaceState()
    disabled.generation = 1
    disabled.enabled = False
    disabled.capture_pending = False
    _MODULE.WorkspaceCmdVelNode._workspace_state_cb(node, disabled)
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK


def test_control_transition_restores_joystick_and_requires_button_rearm():
    node = _authority_node()
    node._authority = _MODULE.AUTHORITY_PATH
    node._orchestrator_state = 'neutral'
    _MODULE.WorkspaceCmdVelNode._orchestrator_state_cb(
        node, String(data='control')
    )
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK
    assert not node._buttons_armed

    # A held path button at the transition cannot seize authority.
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_JOYSTICK
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy())
    assert node._buttons_armed
    _MODULE.WorkspaceCmdVelNode._joy_cb(node, _joy(1))
    assert node._authority == _MODULE.AUTHORITY_PATH

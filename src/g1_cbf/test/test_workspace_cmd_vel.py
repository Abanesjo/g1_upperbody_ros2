import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from g1_cbf_msg.msg import WorkspaceState
from sensor_msgs.msg import Joy
from std_msgs.msg import String


_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'workspace_cmd_vel_node.py'
_SPEC = importlib.util.spec_from_file_location(
    'workspace_cmd_vel_node_script', _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _yaw_error(first, second):
    return abs(_MODULE.wrap_angle(first - second))


def test_interpolated_path_preserves_keyposes_and_resolution():
    samples = _MODULE.interpolate_path(0.05, 5.0)
    cursor = 0
    for x, y, yaw_deg in _MODULE.KEY_POSES:
        expected_yaw = _MODULE.wrap_angle(math.radians(yaw_deg))
        while cursor < len(samples):
            sx, sy, syaw = samples[cursor]
            if (
                np.linalg.norm(np.array([sx - x, sy - y])) < 1e-9
                and _yaw_error(syaw, expected_yaw) < 1e-9
            ):
                break
            cursor += 1
        assert cursor < len(samples)
        cursor += 1

    for previous, current in zip(samples[:-1], samples[1:]):
        distance = np.linalg.norm(
            np.array(current[:2]) - np.array(previous[:2])
        )
        assert distance <= 0.050001
        if distance < 1e-9:
            assert _yaw_error(current[2], previous[2]) <= math.radians(5.001)


def test_follower_drives_straight_then_turns_in_place():
    follower = _MODULE.RectangleFollower()
    velocity, yaw_rate = follower.command(np.array([0.0, 0.0]), 0.0)
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0

    velocity, yaw_rate = follower.command(np.array([0.0, 0.0]), 0.0)
    assert velocity == pytest.approx([0.5, 0.0])
    assert yaw_rate == 0.0

    velocity, _ = follower.command(np.array([0.8, 0.0]), 0.0)
    assert velocity == pytest.approx([0.0, 0.0])
    velocity, yaw_rate = follower.command(np.array([0.8, 0.0]), 0.0)
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate > 0.0


def test_follower_advances_around_unreachable_circle_corner():
    follower = _MODULE.RectangleFollower()
    follower.initial_pose_complete = True
    follower.phase_index = 2
    # A 1.4 m safe radius (2.0 m circle minus configured margins) cannot
    # reach (0.8, 2.0).  This point lies on that safe perimeter and is already
    # closer to the next, horizontal segment than the current vertical one.
    velocity, yaw_rate = follower.command(np.array([0.0, 1.4]), math.pi / 2)
    assert follower.phase_index == 3
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0


def test_follower_does_not_fight_expected_cbf_detour():
    follower = _MODULE.RectangleFollower()
    follower.initial_pose_complete = True
    follower.phase_index = 2
    velocity, yaw_rate = follower.command(
        np.array([0.6, 0.5]),
        math.pi / 2,
        cbf_active=True,
        safe_radius=1.4,
    )
    assert velocity == pytest.approx([0.0, 0.5])
    assert yaw_rate == 0.0


def test_follower_completes_route_with_ideal_circle_cbf_projection():
    follower = _MODULE.RectangleFollower()
    position = np.zeros(2, dtype=np.float64)
    yaw = 0.0
    safe_radius = 1.4
    dt = 0.02
    visited_phases = []

    for _ in range(5000):
        previous_phase = follower.phase_index
        velocity, yaw_rate = follower.command(
            position,
            yaw,
            cbf_active=True,
            safe_radius=safe_radius,
        )

        radius = float(np.linalg.norm(position))
        outward_speed = float(np.dot(position, velocity))
        if radius >= safe_radius - 1e-6 and outward_speed > 0.0:
            velocity -= position * (
                outward_speed / max(float(np.dot(position, position)), 1e-12)
            )

        position += dt * velocity
        radius = float(np.linalg.norm(position))
        if radius > safe_radius:
            position *= safe_radius / radius
        yaw = _MODULE.wrap_angle(yaw + dt * yaw_rate)

        if follower.phase_index != previous_phase:
            visited_phases.append(follower.phase_index)
        if follower.finished:
            break

    assert follower.finished
    assert visited_phases == list(range(1, len(follower.phases) + 1))
    assert np.linalg.norm(position - np.array([0.8, 0.0])) <= 0.12
    assert _yaw_error(yaw, math.pi / 2) <= follower.yaw_tolerance


def test_follower_stops_after_final_pose():
    follower = _MODULE.RectangleFollower()
    follower.initial_pose_complete = True
    follower.phase_index = len(follower.phases) - 1
    velocity, yaw_rate = follower.command(np.array([0.8, 0.0]), math.pi / 2)
    assert follower.finished
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0
    velocity, yaw_rate = follower.command(np.array([0.0, 0.0]), 0.0)
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0


def test_follower_advances_after_joystick_overshoots_endpoint():
    follower = _MODULE.RectangleFollower()
    follower.initial_pose_complete = True
    velocity, yaw_rate = follower.command(np.array([1.2, 0.0]), 0.0)
    assert follower.phase_index == 1
    assert velocity == pytest.approx([0.0, 0.0])
    assert yaw_rate == 0.0


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

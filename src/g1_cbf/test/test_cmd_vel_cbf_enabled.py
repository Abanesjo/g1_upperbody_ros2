import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from geometry_msgs.msg import Twist
from g1_cbf_msg.msg import Capsule, CapsuleArray, WorkspaceState
from rclpy.time import Time


MODULE_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'cmd_vel_cbf_node.py'
)
SPEC = importlib.util.spec_from_file_location('cmd_vel_cbf_node', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CmdVelCBFNode = MODULE.CmdVelCBFNode


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


def test_disabled_cbf_uses_clipped_passthrough_without_state_or_solver():
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    node._workspace_state = (np.array([4.0, -2.0]), True, 2)
    node._latest_cmd = Twist()
    node._latest_cmd.linear.x = 3.0
    node._latest_cmd.linear.y = -2.0
    node._latest_cmd.linear.z = 0.4
    node._latest_cmd.angular.z = 0.7
    node._lin_vel_x_limits = [-1.0, 2.0]
    node._lin_vel_y_limits = [-1.0, 1.0]
    node._cmd_pub = _Publisher()
    node.get_parameter = lambda _name: SimpleNamespace(value=True)

    def _unexpected_call(*_args):
        raise AssertionError('disabled CBF must not inspect state or solve')

    node._state_ready = _unexpected_call
    node._filter_planar_velocity = _unexpected_call

    state = WorkspaceState()
    state.transform.translation.x = 1.5
    state.transform.translation.y = -3.0
    state.enabled = False
    state.generation = 2
    node._workspace_state_cb(state)
    node._tick()

    center_xy, enabled, generation = node._workspace_state
    assert center_xy == pytest.approx([1.5, -3.0])
    assert not enabled
    assert generation == 2
    assert len(node._cmd_pub.messages) == 1
    safe = node._cmd_pub.messages[0]
    assert safe.linear.x == 2.0
    assert safe.linear.y == -1.0
    assert safe.linear.z == 0.4
    assert safe.angular.z == 0.7


def test_reenabled_cbf_returns_to_existing_missing_state_guard():
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    node._workspace_state = (np.zeros(2), False, 0)
    node._latest_cmd = Twist()
    node._latest_cmd.linear.x = 0.8
    node._latest_cmd.angular.z = 0.3
    node._lin_vel_x_limits = [-1.0, 2.0]
    node._lin_vel_y_limits = [-1.0, 1.0]
    node._cmd_pub = _Publisher()
    node.get_parameter = lambda _name: SimpleNamespace(value=True)
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda *_args, **_kwargs: None,
    )

    state_checks = []

    def _missing_state():
        state_checks.append(True)
        return False, 'test state unavailable'

    def _unexpected_filter(_command):
        raise AssertionError('missing state must prevent the CBF solve')

    node._state_ready = _missing_state
    node._filter_planar_velocity = _unexpected_filter

    state = WorkspaceState()
    state.transform.translation.x = 4.0
    state.transform.translation.y = -2.0
    state.enabled = True
    state.generation = 1
    node._workspace_state_cb(state)
    node._tick()

    center_xy, enabled, generation = node._workspace_state
    assert center_xy == pytest.approx([4.0, -2.0])
    assert enabled
    assert generation == 1
    assert state_checks == [True]
    assert len(node._cmd_pub.messages) == 1
    safe = node._cmd_pub.messages[0]
    assert safe.linear.x == 0.0
    assert safe.linear.y == 0.0
    assert safe.angular.z == 0.3


def test_cmd_vel_area_barrier_is_translation_invariant():
    config = MODULE.G1CmdVelCBFConfig.__new__(
        MODULE.G1CmdVelCBFConfig
    )
    config.external_margin_phi = 0.1
    points = jnp.zeros((MODULE.N_HUMAN_ENDPOINT_SPHERES, 2))
    radii = jnp.zeros(MODULE.N_HUMAN_ENDPOINT_SPHERES)
    mask = jnp.zeros(MODULE.N_HUMAN_ENDPOINT_SPHERES, dtype=bool)

    def barrier(head_xy, center_xy):
        return np.asarray(config.h_1(
            jnp.array(head_xy),
            jnp.array([0.0, 0.0, 0.0, 1.0]),
            jnp.array(center_xy),
            jnp.array(3.0),
            jnp.array(0.3),
            jnp.array(True),
            points,
            radii,
            mask,
        ))[0]

    original = barrier([2.0, -1.0], [0.5, -0.25])
    shifted = barrier([7.0, -4.0], [5.5, -3.25])
    assert shifted == pytest.approx(original)


def test_human_capsules_beyond_the_cap_are_dropped_with_a_warning():
    warnings = []
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    node._world_frame = 'world'
    node._tf_pose_lookup = SimpleNamespace(buffer=None)
    node.get_logger = lambda: SimpleNamespace(
        warn=lambda message, **_kwargs: warnings.append(message),
    )
    node.get_clock = lambda: SimpleNamespace(now=lambda: Time(seconds=5.0))
    msg = CapsuleArray()
    msg.header.frame_id = 'world'
    msg.header.stamp = Time(seconds=4.9).to_msg()
    for k in range(MODULE.N_HUMAN_CAPSULES + 3):
        capsule = Capsule()
        capsule.a.x = capsule.b.x = float(k)
        capsule.b.y = 1.0
        capsule.radius = 0.1
        msg.capsules.append(capsule)

    node._human_colliders_cb(msg)

    assert node._human_endpoint_mask.all()
    assert node._human_endpoint_points_xy.shape == (
        MODULE.N_HUMAN_ENDPOINT_SPHERES, 2,
    )
    assert node._human_endpoint_points_xy[-1] == pytest.approx(
        [MODULE.N_HUMAN_CAPSULES - 1, 1.0],
    )
    assert any(
        f'has {MODULE.N_HUMAN_CAPSULES + 3} capsules' in warning
        for warning in warnings
    )


def test_head_state_is_passed_to_solver():
    node = CmdVelCBFNode.__new__(CmdVelCBFNode)
    expected_head_xy = np.array([0.625, -1.125])
    solver_calls = []

    def safety_filter(*args):
        solver_calls.append(np.asarray(args[0], dtype=np.float64))
        return jnp.array([0.2, -0.1])

    node.cbf = SimpleNamespace(safety_filter=safety_filter)
    node._head_xy_world = lambda: expected_head_xy
    node._human_endpoint_args = lambda: (
        np.zeros((MODULE.N_HUMAN_ENDPOINT_SPHERES, 2)),
        np.zeros(MODULE.N_HUMAN_ENDPOINT_SPHERES),
        np.zeros(MODULE.N_HUMAN_ENDPOINT_SPHERES, dtype=bool),
    )
    node._pelvis_quat = np.array([0.0, 0.0, 0.0, 1.0])
    node._world_circle_radius = 1.5
    node._head_collider_radius = 0.3
    node.get_logger = lambda: SimpleNamespace(
        error=lambda *_args, **_kwargs: None,
    )

    safe = node._filter_planar_velocity(
        np.array([0.2, -0.1]),
        np.array([0.0, 0.0]),
    )

    assert safe == pytest.approx([0.2, -0.1])
    assert len(solver_calls) == 1
    assert solver_calls[0] == pytest.approx(expected_head_xy)

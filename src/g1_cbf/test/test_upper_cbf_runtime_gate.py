import importlib.util
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from g1_cbf_msg.msg import WorkspaceState
from rclpy.time import Time
from std_msgs.msg import Bool


_SCRIPT = Path(__file__).parents[1] / 'scripts' / 'g1_cbf_node.py'
_SPEC = importlib.util.spec_from_file_location('g1_cbf_node_script', _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Publisher:
    def __init__(self):
        self.messages = []

    def get_subscription_count(self):
        return 1

    def publish(self, msg):
        self.messages.append(msg)


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message, **_kwargs):
        self.messages.append(message)


def test_disabled_tick_bypasses_qp_and_follows_measured_joints():
    measured = np.linspace(-0.5, 0.5, len(_MODULE.CONTROLLED_JOINTS))
    command_pub = _Publisher()
    active_pairs_pub = _Publisher()
    node = SimpleNamespace(
        q_ctrl=measured,
        q_des_latest=None,
        q_des_filtered=np.ones_like(measured),
        q_cbf_target=np.full_like(measured, 9.0),
        _cbf_enabled=False,
        cmd_pub=command_pub,
        active_pairs_pub=active_pairs_pub,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=12.0)),
    )
    node._publish_empty_active_pairs = MethodType(
        _MODULE.G1CBFNode._publish_empty_active_pairs,
        node,
    )
    node._publish_disabled_command = MethodType(
        _MODULE.G1CBFNode._publish_disabled_command,
        node,
    )

    # The test double intentionally has no CBF solver or parameters. Reaching
    # either would fail, proving the disabled branch returns before the QP.
    _MODULE.G1CBFNode._tick(node)

    assert node.q_cbf_target == pytest.approx(measured)
    assert node.q_cbf_target is not measured
    assert node.q_des_filtered is None
    assert len(command_pub.messages) == 1
    command = command_pub.messages[0]
    assert command.name == list(_MODULE.CONTROLLED_JOINTS)
    assert command.position == pytest.approx(measured)
    assert command.velocity == pytest.approx(np.zeros_like(measured))

    assert len(active_pairs_pub.messages) == 1
    active_pairs = active_pairs_pub.messages[0]
    assert active_pairs.header.frame_id == 'pelvis'
    assert not active_pairs.robot_body_index
    assert not active_pairs.internal_body_a_index


def test_required_missing_localization_holds_before_filter_or_qp():
    measured = np.linspace(-0.4, 0.4, len(_MODULE.CONTROLLED_JOINTS))
    command_pub = _Publisher()
    active_pairs_pub = _Publisher()
    node = SimpleNamespace(
        q_ctrl=measured,
        q_des_latest=np.full_like(measured, 0.8),
        q_des_filtered=np.full_like(measured, 0.7),
        q_cbf_target=np.full_like(measured, 0.6),
        _cbf_enabled=True,
        _workspace_state=(np.zeros(2), True, 1),
        cmd_pub=command_pub,
        active_pairs_pub=active_pairs_pub,
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=12.0)),
        _active_human_capsules=lambda: [],
        _needs_pelvis_pose=lambda _humans, _workspace: True,
        _lookup_pelvis_pose=lambda: None,
    )
    node._publish_empty_active_pairs = MethodType(
        _MODULE.G1CBFNode._publish_empty_active_pairs,
        node,
    )
    node._publish_disabled_command = MethodType(
        _MODULE.G1CBFNode._publish_disabled_command,
        node,
    )
    node._publish_localization_hold_command = MethodType(
        _MODULE.G1CBFNode._publish_localization_hold_command,
        node,
    )

    # There is deliberately no parameter accessor or CBF solver. The missing
    # required pose must return before either command filtering or the QP.
    _MODULE.G1CBFNode._tick(node)

    assert node.q_cbf_target == pytest.approx(measured)
    assert node.q_cbf_target is not measured
    assert node.q_des_filtered is None
    assert len(command_pub.messages) == 1
    command = command_pub.messages[0]
    assert command.position == pytest.approx(measured)
    assert command.velocity == pytest.approx(np.zeros_like(measured))
    assert len(active_pairs_pub.messages) == 1


@pytest.mark.parametrize(
    ('pose', 'age_sec', 'expected_pose', 'message_fragment'),
    [
        (None, None, None, 'TF lookup failed'),
        (
            SimpleNamespace(
                position=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            0.501,
            None,
            'is stale by 0.501s',
        ),
        (
            SimpleNamespace(
                position=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            math.inf,
            None,
            'is future-dated',
        ),
        (
            SimpleNamespace(
                position=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            0.5,
            'same',
            None,
        ),
        (
            SimpleNamespace(
                position=np.zeros(3),
                quat=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            None,
            'same',
            None,
        ),
    ],
    ids=['missing', 'stale', 'future', 'fresh-boundary', 'zero-stamp'],
)
def test_pelvis_pose_lookup_rejects_unusable_localization(
        pose, age_sec, expected_pose, message_fragment):
    logger = _Logger()

    class _TfLookup:
        def lookup(self):
            return pose, 'not connected' if pose is None else ''

        def age_sec(self, _pose):
            return age_sec

        def describe(self):
            return 'world -> pelvis'

    node = SimpleNamespace(
        _tf_pose_lookup=_TfLookup(),
        _tf_stale_timeout_sec=0.5,
        get_logger=lambda: logger,
    )

    result = _MODULE.G1CBFNode._lookup_pelvis_pose(node)

    if expected_pose == 'same':
        assert result is pose
    else:
        assert result is expected_pose
    if message_fragment is None:
        assert logger.messages == []
    else:
        assert any(message_fragment in message for message in logger.messages)


def test_pelvis_pose_lookup_rejects_nonfinite_pose_and_disabled_age_gate():
    logger = _Logger()
    valid_pose = SimpleNamespace(
        position=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )

    class _TfLookup:
        def __init__(self, pose):
            self.pose = pose

        def lookup(self):
            return self.pose, ''

        def age_sec(self, _pose):
            return 10.0

        def describe(self):
            return 'world -> pelvis'

    node = SimpleNamespace(
        _tf_pose_lookup=_TfLookup(valid_pose),
        _tf_stale_timeout_sec=0.0,
        get_logger=lambda: logger,
    )
    assert _MODULE.G1CBFNode._lookup_pelvis_pose(node) is valid_pose

    invalid_pose = SimpleNamespace(
        position=np.array([np.nan, 0.0, 0.0]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    node._tf_pose_lookup = _TfLookup(invalid_pose)
    assert _MODULE.G1CBFNode._lookup_pelvis_pose(node) is None
    assert any('non-finite' in message for message in logger.messages)


def test_tf_pose_future_stamp_maps_to_infinite_age_for_fail_closed_gates():
    lookup = _MODULE.TfPoseLookup.__new__(_MODULE.TfPoseLookup)
    lookup._node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(
            now=lambda: Time(seconds=10.0),
        ),
    )

    zero_stamp_pose = SimpleNamespace(stamp=Time())
    future_pose = SimpleNamespace(stamp=Time(seconds=10.1))
    past_pose = SimpleNamespace(stamp=Time(seconds=9.75))

    assert lookup.age_sec(zero_stamp_pose) is None
    assert math.isinf(lookup.age_sec(future_pose))
    assert lookup.age_sec(past_pose) == pytest.approx(0.25)


def test_reenable_resets_filter_target_to_current_measurement():
    measured = np.linspace(-0.25, 0.25, len(_MODULE.CONTROLLED_JOINTS))
    latest_unsafe = np.linspace(0.5, 1.0, len(_MODULE.CONTROLLED_JOINTS))
    logger = _Logger()
    node = SimpleNamespace(
        _cbf_enabled=False,
        q_ctrl=measured,
        q_cbf_target=np.full_like(measured, 9.0),
        q_des_filtered=np.full_like(measured, 8.0),
        q_des_latest=latest_unsafe,
        get_logger=lambda: logger,
    )

    _MODULE.G1CBFNode._cbf_enabled_cb(node, Bool(data=True))

    assert node._cbf_enabled
    assert node.q_cbf_target == pytest.approx(measured)
    assert node.q_cbf_target is not measured
    assert node.q_des_filtered is None
    assert node.q_des_latest is latest_unsafe
    assert logger.messages == ['CBF safety filter enabled']


def test_external_reenable_reseeds_without_changing_overall_gate_or_command():
    measured = np.linspace(-0.3, 0.3, len(_MODULE.CONTROLLED_JOINTS))
    latest_unsafe = np.linspace(0.4, 0.9, len(_MODULE.CONTROLLED_JOINTS))
    logger = _Logger()
    node = SimpleNamespace(
        _cbf_enabled=True,
        _external_enabled=False,
        q_ctrl=measured,
        q_cbf_target=np.full_like(measured, 9.0),
        q_des_filtered=np.full_like(measured, 8.0),
        q_des_latest=latest_unsafe,
        get_logger=lambda: logger,
    )

    _MODULE.G1CBFNode._external_enabled_cb(node, Bool(data=True))

    assert node._external_enabled
    assert node._cbf_enabled
    assert node.q_cbf_target == pytest.approx(measured)
    assert node.q_cbf_target is not measured
    assert node.q_des_filtered is None
    assert node.q_des_latest is latest_unsafe
    assert logger.messages == ['External human-collision CBF enabled']


def test_external_disable_masks_cached_humans_without_mutating_other_targets():
    cached_humans = [{
        'a': np.array([1.0, 0.0, 0.0]),
        'b': np.array([1.0, 0.0, 1.0]),
        'radius': 0.2,
    }]
    q_target = np.linspace(-0.2, 0.2, len(_MODULE.CONTROLLED_JOINTS))
    q_filtered = np.linspace(0.1, 0.4, len(_MODULE.CONTROLLED_JOINTS))
    q_latest = np.linspace(0.3, 0.8, len(_MODULE.CONTROLLED_JOINTS))
    logger = _Logger()
    freshness_calls = []
    node = SimpleNamespace(
        _cbf_enabled=True,
        _external_enabled=True,
        _human_capsules=cached_humans,
        q_ctrl=np.zeros_like(q_target),
        q_cbf_target=q_target,
        q_des_filtered=q_filtered,
        q_des_latest=q_latest,
        get_logger=lambda: logger,
    )

    def current_humans():
        freshness_calls.append(True)
        return cached_humans

    node._current_human_capsules = current_humans
    _MODULE.G1CBFNode._external_enabled_cb(node, Bool(data=False))
    active = _MODULE.G1CBFNode._active_human_capsules(node)

    assert not node._external_enabled
    assert node._cbf_enabled
    assert active == []
    assert freshness_calls == []
    assert node._human_capsules is cached_humans
    assert node.q_cbf_target is q_target
    assert node.q_des_filtered is q_filtered
    assert node.q_des_latest is q_latest
    assert logger.messages == ['External human-collision CBF disabled']


def test_external_disabled_tick_keeps_internal_cbf_and_solver_active():
    joint_count = len(_MODULE.CONTROLLED_JOINTS)
    cached_humans = [{
        'a': np.array([1.0, 0.0, 0.0]),
        'b': np.array([1.0, 0.0, 1.0]),
        'radius': 0.2,
    }]
    calls = {
        'pack': [],
        'external_selector': 0,
        'internal_selector': 0,
        'solver': 0,
    }
    command_pub = _Publisher()

    class FakeCBF:
        def safety_filter(self, *args):
            calls['solver'] += 1
            assert not np.asarray(args[6]).any()
            return _MODULE.jnp.zeros(joint_count, dtype=_MODULE.jnp.float64)

    def pack_humans(pelvis_pose, humans):
        calls['pack'].append((pelvis_pose, humans))
        return (
            _MODULE.jnp.zeros(
                (_MODULE.N_HUMAN_CAPSULES, 7), dtype=_MODULE.jnp.float64,
            ),
            _MODULE.jnp.zeros(_MODULE.N_HUMAN_CAPSULES, dtype=bool),
        )

    def select_external(_z, _q_legs, _capsules, human_mask):
        calls['external_selector'] += 1
        assert not np.asarray(human_mask).any()
        return (
            _MODULE.jnp.zeros((1, 2), dtype=_MODULE.jnp.int32),
            _MODULE.jnp.zeros(1, dtype=bool),
            _MODULE.jnp.full(1, _MODULE.jnp.inf, dtype=_MODULE.jnp.float64),
        )

    def select_internal(_z, _q_legs):
        calls['internal_selector'] += 1
        return (
            _MODULE.jnp.zeros((1, 4), dtype=_MODULE.jnp.int32),
            _MODULE.jnp.ones(1, dtype=bool),
            _MODULE.jnp.zeros(1, dtype=_MODULE.jnp.float64),
        )

    parameters = {
        'dt': 0.01,
        'K': 1.0,
        'max_velocity': 1.0,
        'lpf_gain': 1.0,
        'evaluate_at_actual': True,
        'max_lead': 0.5,
    }
    node = SimpleNamespace(
        q_ctrl=np.zeros(joint_count),
        q_legs=np.zeros(_MODULE.N_LEG_JOINTS),
        q_des_latest=np.full(joint_count, 0.25),
        q_des_filtered=None,
        q_cbf_target=np.zeros(joint_count),
        _cbf_enabled=True,
        _external_enabled=False,
        _human_capsules=cached_humans,
        _workspace_state=(np.zeros(2), False, 0),
        cbf=FakeCBF(),
        cmd_pub=command_pub,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
        get_clock=lambda: SimpleNamespace(now=lambda: Time(seconds=7.0)),
        _current_human_capsules=lambda: pytest.fail(
            'disabled external CBF inspected the human cache'
        ),
        _needs_pelvis_pose=lambda humans, _workspace: bool(humans),
        _lookup_pelvis_pose=lambda: pytest.fail(
            'empty active humans unexpectedly requested pelvis TF'
        ),
        _pack_human_capsules=pack_humans,
        _select_active_external_pairs=select_external,
        _select_active_internal_pairs=select_internal,
        _head_circle_args=lambda _pose, _workspace: (
            _MODULE.jnp.zeros(3, dtype=_MODULE.jnp.float64),
            _MODULE.jnp.array(
                [0.0, 0.0, 0.0, 1.0], dtype=_MODULE.jnp.float64,
            ),
            _MODULE.jnp.zeros(2, dtype=_MODULE.jnp.float64),
            _MODULE.jnp.array(3.0, dtype=_MODULE.jnp.float64),
            _MODULE.jnp.array(0.3, dtype=_MODULE.jnp.float64),
            _MODULE.jnp.array(False, dtype=bool),
        ),
        _publish_active_pairs=lambda *_args: None,
    )
    node._active_human_capsules = MethodType(
        _MODULE.G1CBFNode._active_human_capsules,
        node,
    )

    _MODULE.G1CBFNode._tick(node)

    assert calls['pack'] == [(None, [])]
    assert calls['external_selector'] == 1
    assert calls['internal_selector'] == 1
    assert calls['solver'] == 1
    assert node._human_capsules is cached_humans
    assert len(command_pub.messages) == 1


def test_workspace_state_does_not_disable_upper_cbf():
    node = SimpleNamespace(
        _cbf_enabled=True,
        _workspace_state=(np.zeros(2), True, 0),
        get_parameter=lambda _name: SimpleNamespace(value=True),
    )
    state = WorkspaceState()
    state.transform.translation.x = 2.0
    state.transform.translation.y = -1.0
    state.enabled = False
    state.generation = 3

    _MODULE.G1CBFNode._workspace_state_cb(node, state)
    args = _MODULE.G1CBFNode._head_circle_args(
        node,
        None,
        node._workspace_state,
    )

    assert node._cbf_enabled
    assert np.asarray(args[2]) == pytest.approx([2.0, -1.0])
    assert not bool(np.asarray(args[-1]))


def test_workspace_head_circle_remains_enabled_when_external_cbf_is_disabled():
    pelvis_pose = SimpleNamespace(
        position=np.array([1.0, -2.0, 0.8]),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    parameters = {
        'area_cbf': True,
        'head_circle_cbf_enabled': True,
        'world_circle_radius': 3.0,
        'head_collider_radius': 0.3,
    }
    node = SimpleNamespace(
        _external_enabled=False,
        get_parameter=lambda name: SimpleNamespace(value=parameters[name]),
    )

    args = _MODULE.G1CBFNode._head_circle_args(
        node,
        pelvis_pose,
        (np.array([0.5, -0.25]), True, 4),
    )

    assert not node._external_enabled
    assert np.asarray(args[2]) == pytest.approx([0.5, -0.25])
    assert bool(np.asarray(args[-1]))


def test_upper_area_barrier_is_translation_invariant():
    config = _MODULE.G1CollisionCBFConfig.__new__(
        _MODULE.G1CollisionCBFConfig
    )
    config.external_margin_phi = 0.1
    identity = _MODULE.jnp.array([0.0, 0.0, 0.0, 1.0])

    def barrier(pelvis_xy, center_xy):
        return config._head_circle_barrier(
            _MODULE.jnp.array([0.2, -0.1, 0.8]),
            _MODULE.jnp.array([pelvis_xy[0], pelvis_xy[1], 1.0]),
            identity,
            _MODULE.jnp.array(center_xy),
            _MODULE.jnp.array(3.0),
            _MODULE.jnp.array(0.3),
            _MODULE.jnp.array(True),
        )

    original = barrier([1.0, 2.0], [0.5, 1.5])
    shifted = barrier([4.0, -2.0], [3.5, -2.5])
    assert float(shifted) == pytest.approx(float(original))

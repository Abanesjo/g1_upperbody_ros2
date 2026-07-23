import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / 'scripts' / 'workspace_manager_node.py'
)
SPEC = importlib.util.spec_from_file_location(
    'workspace_manager_node',
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
WorkspaceCaptureState = MODULE.WorkspaceCaptureState
workspace_transform = MODULE.workspace_transform


def _pose(x=1.0, y=2.0, z=0.8, quat=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=[x, y, z],
        quat=list(quat),
    )


def test_initial_state_is_disabled_identity_generation_zero():
    state = WorkspaceCaptureState()
    transform = workspace_transform(state)

    assert not state.enabled
    assert not state.pending
    assert state.generation == 0
    assert transform.translation.x == 0.0
    assert transform.translation.y == 0.0
    assert transform.translation.z == 0.0
    assert transform.rotation.x == 0.0
    assert transform.rotation.y == 0.0
    assert transform.rotation.z == 0.0
    assert transform.rotation.w == 1.0


def test_capture_uses_only_pelvis_xy_and_increments_generation():
    state = WorkspaceCaptureState()
    state.request_enable(now=10.0, retry_timeout_sec=0.5)

    captured, reason = state.try_capture(
        _pose(x=3.5, y=-1.25, z=9.0, quat=(0.2, 0.3, 0.4, 0.5)),
        age_sec=0.1,
        stale_timeout_sec=0.2,
    )

    assert captured, reason
    assert state.enabled
    assert not state.pending
    assert state.generation == 1
    transform = workspace_transform(state)
    assert transform.translation.x == 3.5
    assert transform.translation.y == -1.25
    assert transform.translation.z == 0.0
    assert transform.rotation.w == 1.0

    state.request_disable()
    assert not state.enabled
    assert (state.x, state.y, state.generation) == (3.5, -1.25, 1)

    state.request_enable(now=20.0, retry_timeout_sec=0.5)
    captured, reason = state.try_capture(
        _pose(x=-2.0, y=4.0),
        age_sec=None,
        stale_timeout_sec=0.2,
    )
    assert captured, reason
    assert (state.x, state.y, state.generation) == (-2.0, 4.0, 2)


@pytest.mark.parametrize(
    ('pose', 'age_sec', 'expected_reason'),
    [
        (_pose(x=math.nan), 0.0, 'non-finite'),
        (_pose(), math.nan, 'invalid timestamp'),
        (_pose(), -0.1, 'invalid timestamp'),
        (_pose(), 0.21, 'stale'),
    ],
)
def test_invalid_or_stale_pose_is_retried_without_changing_transform(
    pose,
    age_sec,
    expected_reason,
):
    state = WorkspaceCaptureState(x=7.0, y=8.0, enabled=True, generation=4)
    state.request_enable(now=10.0, retry_timeout_sec=0.5)

    captured, reason = state.try_capture(
        pose,
        age_sec=age_sec,
        stale_timeout_sec=0.2,
    )

    assert not captured
    assert expected_reason in reason
    assert state.pending
    assert not state.enabled
    assert (state.x, state.y, state.generation) == (7.0, 8.0, 4)


def test_timeout_requires_a_new_true_request_and_false_cancels():
    state = WorkspaceCaptureState(x=7.0, y=8.0, enabled=True, generation=4)
    state.request_enable(now=10.0, retry_timeout_sec=0.5)

    assert not state.expire(10.49)
    assert state.pending
    assert state.expire(10.5)
    assert not state.pending
    assert not state.enabled
    assert (state.x, state.y, state.generation) == (7.0, 8.0, 4)

    captured, reason = state.try_capture(
        _pose(x=1.0, y=1.0),
        age_sec=0.0,
        stale_timeout_sec=0.2,
    )
    assert not captured
    assert reason == 'no workspace capture is pending'

    state.request_enable(now=11.0, retry_timeout_sec=0.5)
    state.request_disable()
    assert not state.pending
    assert not state.enabled

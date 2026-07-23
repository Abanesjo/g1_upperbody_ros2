from importlib.util import module_from_spec
from importlib.util import spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "motion_playback.py"
SPEC = spec_from_file_location("motion_playback", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _buttons(*pressed):
    buttons = [0] * 11
    for index in pressed:
        buttons[index] = 1
    return buttons


def _release(gate):
    assert gate.handle_joy(_buttons()) is False


def test_startup_is_inactive_and_initial_state_publishes_nothing():
    gate = MODULE.MotionPlaybackGate()

    assert gate.active is False
    assert gate.next_frame(4) is None
    assert gate.handle_orchestrator_state("neutral") is False
    assert gate.frame_idx == 0


def test_start_requires_fresh_button_edge_in_control():
    gate = MODULE.MotionPlaybackGate()

    assert gate.handle_joy(_buttons(MODULE.START_BUTTON)) is False
    assert gate.active is False
    assert gate.handle_orchestrator_state("control") is False
    assert gate.handle_joy(_buttons(MODULE.START_BUTTON)) is False
    assert gate.active is False

    _release(gate)
    assert gate.handle_joy(_buttons(MODULE.START_BUTTON)) is False
    assert gate.active is True
    assert gate.next_frame(3) == 0


def test_start_restarts_active_playback_from_frame_zero():
    gate = MODULE.MotionPlaybackGate()
    gate.handle_orchestrator_state("control")
    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.next_frame(3) == 0
    assert gate.next_frame(3) == 1

    _release(gate)
    assert gate.handle_joy(_buttons(MODULE.START_BUTTON)) is False
    assert gate.next_frame(3) == 0


def test_stop_wins_and_requests_one_default_target_per_edge():
    gate = MODULE.MotionPlaybackGate()
    gate.handle_orchestrator_state("control")
    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.next_frame(3) == 0

    assert gate.handle_joy(
        _buttons(MODULE.STOP_BUTTON, MODULE.START_BUTTON)
    ) is True
    assert gate.active is False
    assert gate.frame_idx == 0
    assert gate.handle_joy(
        _buttons(MODULE.STOP_BUTTON, MODULE.START_BUTTON)
    ) is False

    _release(gate)
    assert gate.handle_joy(_buttons(MODULE.STOP_BUTTON)) is True
    assert gate.active is False


def test_control_exit_stops_active_motion_once_and_resets():
    gate = MODULE.MotionPlaybackGate()
    gate.handle_orchestrator_state("control")
    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.next_frame(3) == 0
    assert gate.next_frame(3) == 1

    assert gate.handle_orchestrator_state("neutral") is True
    assert gate.active is False
    assert gate.frame_idx == 0
    assert gate.handle_orchestrator_state("neutral") is False
    assert gate.handle_orchestrator_state("damp") is False


def test_control_entry_resets_progress_and_requires_fresh_start():
    gate = MODULE.MotionPlaybackGate()
    gate.handle_orchestrator_state("control")
    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.next_frame(3) == 0

    assert gate.handle_orchestrator_state("neutral") is True
    assert gate.handle_orchestrator_state("control") is False
    assert gate.active is False
    assert gate.frame_idx == 0
    assert gate.handle_joy(_buttons(MODULE.START_BUTTON)) is False
    assert gate.active is False

    _release(gate)
    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.next_frame(3) == 0


def test_looping_and_non_looping_frame_progression():
    looping = MODULE.MotionPlaybackGate(loop=True)
    looping.handle_orchestrator_state("control")
    looping.handle_joy(_buttons(MODULE.START_BUTTON))
    assert [looping.next_frame(2) for _ in range(5)] == [0, 1, 0, 1, 0]

    one_shot = MODULE.MotionPlaybackGate(
        loop=False, orchestrator_required=False
    )
    one_shot.handle_joy(_buttons(MODULE.START_BUTTON))
    assert one_shot.next_frame(2) == 0
    assert one_shot.next_frame(2) == 1
    assert one_shot.next_frame(2) is None
    assert one_shot.active is False
    assert one_shot.frame_idx == 0


def test_orchestrator_can_be_optional():
    gate = MODULE.MotionPlaybackGate(orchestrator_required=False)

    gate.handle_joy(_buttons(MODULE.START_BUTTON))
    assert gate.active is True
    assert gate.next_frame(2) == 0
    assert gate.handle_orchestrator_state("neutral") is False
    assert gate.active is True


def test_default_target_matches_joint_order():
    assert len(MODULE.UPPER_BODY_JOINTS) == 11
    assert MODULE.DEFAULT_UPPER_BODY_TARGET == [
        0.0,
        0.0,
        0.0,
        0.37,
        0.62,
        0.0,
        0.82,
        0.33,
        -0.67,
        0.0,
        1.01,
    ]

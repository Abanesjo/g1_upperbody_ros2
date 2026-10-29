import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions


_LAUNCH = Path(__file__).parents[1] / 'launch' / 'g1_human_manual.launch.py'
_SPEC = importlib.util.spec_from_file_location('g1_human_manual_launch', _LAUNCH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_launch_arguments_and_defaults():
    context = LaunchContext()
    arguments = {
        entity.name: perform_substitutions(context, entity.default_value)
        for entity in _MODULE.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert arguments == {
        'use_sim_time': 'false',
        'num_humans': '1',
        'human_x': '0.5',
        'human_y': '0.0',
        'human_yaw_deg': '0.0',
        'human_spacing': '1.0',
    }


def test_one_human_keeps_the_previous_default_pose():
    assert _MODULE.human_layout(1, 0.5, 0.0, 0.0, 1.0) == [
        ('human_0', 0.5, 0.0, 0.0),
    ]


def test_humans_form_a_row_centred_on_the_given_position():
    layout = _MODULE.human_layout(3, 0.5, 0.2, 90.0, 1.0)

    assert [namespace for namespace, *_ in layout] == [
        'human_0', 'human_1', 'human_2',
    ]
    assert [y for _, _, y, _ in layout] == pytest.approx([-0.8, 0.2, 1.2])
    assert all(x == 0.5 and yaw == 90.0 for _, x, _, yaw in layout)


def test_fewer_than_one_human_is_rejected():
    with pytest.raises(ValueError):
        _MODULE.human_layout(0, 0.5, 0.0, 0.0, 1.0)

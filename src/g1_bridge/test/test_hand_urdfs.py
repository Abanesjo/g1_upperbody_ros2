import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


DESCRIPTION_ROOT = Path(__file__).parents[2] / 'g1_description'
NUM_MOTORS = 29
MESH_PREFIX = 'package://g1_description/'

# hand_type -> (left palm link, right palm link, expected mass per hand [kg])
HANDS = {
    'inspire': ('L_hand_base_link', 'R_hand_base_link', 0.192),
    'dex': ('left_hand_palm_link', 'right_hand_palm_link', 0.697),
}


def _load(hand_type):
    return ET.parse(
        DESCRIPTION_ROOT / 'urdf' / f'g1_29_{hand_type}.urdf').getroot()


def _subtree_mass(robot, root_link):
    children = {}
    for joint in robot.findall('joint'):
        children.setdefault(
            joint.find('parent').attrib['link'], []
        ).append(joint.find('child').attrib['link'])
    masses = {
        link.attrib['name']: float(link.find('inertial/mass').attrib['value'])
        for link in robot.findall('link')
        if link.find('inertial') is not None
    }
    total = 0.0
    pending = [root_link]
    while pending:
        link = pending.pop()
        total += masses.get(link, 0.0)
        pending.extend(children.get(link, []))
    return total


@pytest.mark.parametrize('hand_type', sorted(HANDS))
def test_only_body_joints_are_actuated(hand_type):
    robot = _load(hand_type)
    moving = [
        joint.attrib['name'] for joint in robot.findall('joint')
        if joint.attrib['type'] != 'fixed'
    ]

    assert len(moving) == NUM_MOTORS
    assert not [name for name in moving if 'hand' in name or name[:2] in ('L_', 'R_')]


@pytest.mark.parametrize('hand_type', sorted(HANDS))
def test_hand_mass(hand_type):
    robot = _load(hand_type)
    left, right, expected = HANDS[hand_type]

    assert math.isclose(_subtree_mass(robot, left), expected, abs_tol=2e-3)
    assert math.isclose(_subtree_mass(robot, right), expected, abs_tol=2e-3)


@pytest.mark.parametrize('hand_type', sorted(HANDS))
def test_pelvis_is_root_and_meshes_exist(hand_type):
    robot = _load(hand_type)
    links = {link.attrib['name'] for link in robot.findall('link')}
    child_links = {
        joint.find('child').attrib['link'] for joint in robot.findall('joint')
    }

    assert links - child_links == {'pelvis'}
    for mesh in robot.iter('mesh'):
        filename = mesh.attrib['filename']
        assert filename.startswith(MESH_PREFIX)
        assert (DESCRIPTION_ROOT / filename[len(MESH_PREFIX):]).is_file()

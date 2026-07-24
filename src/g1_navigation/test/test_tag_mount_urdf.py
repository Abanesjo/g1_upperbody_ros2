from pathlib import Path
import xml.etree.ElementTree as ET


SRC_ROOT = Path(__file__).parents[2]
URDF_PATH = SRC_ROOT / 'g1_description' / 'urdf' / 'g1_29_inspire.urdf'
RVIZ_PATH = SRC_ROOT / 'g1_bridge' / 'rviz' / 'config.rviz'


def _joint_map(robot):
    return {
        joint.find('child').attrib['link']: joint
        for joint in robot.findall('joint')
    }


def test_pelvis_remains_root_and_tag_mount_has_corrected_torso_mount():
    robot = ET.parse(URDF_PATH).getroot()
    links = {link.attrib['name'] for link in robot.findall('link')}
    joints_by_child = _joint_map(robot)
    roots = links - set(joints_by_child)

    assert roots == {'pelvis'}
    assert 'tag_frame' not in links
    assert 'tag_mount_frame' in links

    mount_joint = joints_by_child['tag_mount_frame']
    assert mount_joint.attrib == {
        'name': 'imu_in_torso_to_tag_mount_frame',
        'type': 'fixed',
    }
    assert mount_joint.find('parent').attrib['link'] == 'imu_in_torso'
    assert mount_joint.find('origin').attrib == {
        'xyz': '0.05 0 0',
        # Original mounting orientation post-rotated 180 degrees around the
        # tag frame's local Y axis to correct its front/back direction.
        'rpy': '-1.5707963267948966 0 -1.5707963267948966',
    }


def test_tag_mount_path_retains_all_dynamic_waist_joints():
    robot = ET.parse(URDF_PATH).getroot()
    joints_by_child = _joint_map(robot)
    path = []
    child = 'tag_mount_frame'
    while child in joints_by_child:
        joint = joints_by_child[child]
        parent = joint.find('parent').attrib['link']
        path.append(
            (joint.attrib['name'], parent, child, joint.attrib['type'])
        )
        child = parent

    assert path == [
        (
            'imu_in_torso_to_tag_mount_frame',
            'imu_in_torso',
            'tag_mount_frame',
            'fixed',
        ),
        (
            'imu_in_torso_joint',
            'torso_link',
            'imu_in_torso',
            'fixed',
        ),
        (
            'waist_pitch_joint',
            'waist_roll_link',
            'torso_link',
            'revolute',
        ),
        (
            'waist_roll_joint',
            'waist_yaw_link',
            'waist_roll_link',
            'revolute',
        ),
        (
            'waist_yaw_joint',
            'pelvis',
            'waist_yaw_link',
            'revolute',
        ),
    ]


def test_rviz_distinguishes_detected_tag_from_modeled_mount():
    config = RVIZ_PATH.read_text()

    assert 'tag_mount_frame:' in config
    assert 'ghost/tag_mount_frame:' in config
    assert 'tag_frame:' in config
    assert 'imu_in_torso:\n                      tag_frame:' not in config

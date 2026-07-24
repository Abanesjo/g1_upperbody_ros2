from pathlib import Path
import os
import shlex
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
APRILTAG_LAUNCH = PACKAGE_ROOT / 'launch' / 'apriltag_localization.launch.xml'
ODOM_LAUNCH = PACKAGE_ROOT / 'launch' / 'odom_simulation.launch.xml'
ODOM_NODE = (
    PACKAGE_ROOT / 'scripts' / 'base_footprint_tf_publisher_node.py'
)
APRILTAG_NODE = (
    PACKAGE_ROOT / 'scripts' / 'apriltag_localization_node.py'
)
CMAKE_PATH = PACKAGE_ROOT / 'CMakeLists.txt'


def _arguments(root):
    return {
        argument.attrib['name']: argument.attrib.get('default')
        for argument in root.findall('arg')
    }


def _parameters(node):
    return {
        parameter.attrib['name']: parameter.attrib.get('value')
        for parameter in node.findall('param')
    }


def _static_frames(node):
    tokens = shlex.split(node.attrib['args'])
    parent = tokens[tokens.index('--frame-id') + 1]
    child = tokens[tokens.index('--child-frame-id') + 1]
    return parent, child


def test_apriltag_launch_exposes_exact_time_localization_contract():
    root = ET.parse(APRILTAG_LAUNCH).getroot()

    assert _arguments(root) == {
        'tag_pose_topic': '/pose/tag_frame',
        'pelvis_pose_topic': '/pose/pelvis',
        'odom_topic': '/odom',
        'diagnostics_topic': '/diagnostics',
        'world_frame': 'world',
        'detected_tag_frame': 'tag_frame',
        'tag_mount_frame': 'tag_mount_frame',
        'pelvis_frame': 'pelvis',
        'queue_depth': '20',
        'kinematic_wait_timeout_sec': '0.20',
        'source_stale_timeout_sec': '0.50',
        'future_tolerance_sec': '0.05',
        'tf_lookup_timeout_sec': '0.0',
        'retry_rate_hz': '100.0',
        'diagnostic_rate_hz': '2.0',
        'pose_position_stddev_m': '0.05',
        'pose_orientation_stddev_rad': '0.08726646259971647',
        'twist_linear_stddev_mps': '1.0',
        'twist_angular_stddev_radps': '1.0',
        'twist_min_dt_sec': '0.02',
        'twist_max_dt_sec': '0.50',
        'twist_max_linear_speed_mps': '3.0',
        'twist_max_angular_speed_radps': '4.0',
        'twist_max_translation_step_m': '0.50',
        'twist_max_rotation_step_rad': '0.7853981633974483',
        'use_sim_time': 'false',
    }
    nodes = root.findall('node')
    assert len(nodes) == 1
    node = nodes[0]
    assert node.attrib == {
        'pkg': 'g1_navigation',
        'exec': 'apriltag_localization_node',
        'name': 'apriltag_localization_node',
        'output': 'screen',
    }
    assert _parameters(node) == {
        name: f'$(var {name})'
        for name in _arguments(root)
    }
    assert not root.findall('include')
    source = APRILTAG_LAUNCH.read_text()
    assert 'livox' not in source.lower()
    assert 'publish_rate_hz' not in source


def test_apriltag_executable_is_installable_and_thin():
    assert os.access(APRILTAG_NODE, os.X_OK)
    assert APRILTAG_NODE.read_text().splitlines()[-2:] == [
        "if __name__ == '__main__':",
        '    main()',
    ]
    cmake = CMAKE_PATH.read_text()
    assert 'scripts/apriltag_localization_node.py' in cmake
    assert 'RENAME apriltag_localization_node' in cmake


def test_odom_simulation_keeps_its_independent_world_base_pelvis_tree():
    root = ET.parse(ODOM_LAUNCH).getroot()
    nodes = root.findall('node')
    executables = {node.attrib['exec'] for node in nodes}

    assert 'base_footprint_tf_publisher_node' in executables
    assert 'apriltag_localization_node' not in executables
    static_frames = {
        _static_frames(node)
        for node in nodes
        if node.attrib['exec'] == 'static_transform_publisher'
    }
    assert ('base_footprint', 'pelvis') in static_frames
    assert ('mid360_link', 'livox_frame') in static_frames
    assert 'tag_frame' not in ODOM_LAUNCH.read_text()

    node_source = ODOM_NODE.read_text()
    assert "transform.header.frame_id = 'world'" in node_source
    assert "transform.child_frame_id = 'base_footprint'" in node_source
    assert "odom.header.frame_id = 'world'" in node_source
    assert "odom.child_frame_id = 'base_footprint'" in node_source
    assert 'except KeyboardInterrupt:' in node_source
    assert 'if rclpy.ok():' in node_source

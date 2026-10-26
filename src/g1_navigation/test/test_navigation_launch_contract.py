from pathlib import Path
import os
import shlex
import xml.etree.ElementTree as ET


PACKAGE_ROOT = Path(__file__).parents[1]
APRILTAG_LAUNCH = PACKAGE_ROOT / 'launch' / 'apriltag_localization.launch.xml'
ODOM_LAUNCH = PACKAGE_ROOT / 'launch' / 'odom_simulation.launch.xml'
VICON_LAUNCH = PACKAGE_ROOT / 'launch' / 'vicon.launch.xml'
ODOM_NODE = (
    PACKAGE_ROOT / 'scripts' / 'base_footprint_tf_publisher_node.py'
)
APRILTAG_NODE = (
    PACKAGE_ROOT / 'scripts' / 'apriltag_localization_node.py'
)
ANCHOR_NODE = (
    PACKAGE_ROOT / 'scripts' / 'fusion_world_anchor_node.py'
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


def test_vicon_launch_localizes_pelvis_through_torso_mount():
    root = ET.parse(VICON_LAUNCH).getroot()
    args = _arguments(root)
    assert args['vicon_pose_topic'] == '/vicon/unitree_g1/unitree_g1'
    assert args['world_frame'] == 'world'
    assert args['pelvis_frame'] == 'pelvis'
    assert args['mount_parent_frame'] == 'torso_link'
    assert args['use_sim_time'] == 'false'

    static = root.find('node')
    assert static.attrib['exec'] == 'static_transform_publisher'
    assert static.attrib['args'].endswith(
        '--frame-id $(var mount_parent_frame) '
        '--child-frame-id $(var mount_frame)')

    includes = {
        Path(include.attrib['file']).name: {
            arg.attrib['name']: arg.attrib['value']
            for arg in include.findall('arg')
        }
        for include in root.iter('include')
    }
    assert includes['client.launch.py']['world_frame'] == (
        '$(var world_frame)')
    # Occluded subjects are NaN transforms; keep them off the shared /tf.
    remap = root.find('group/set_remap')
    assert remap.attrib == {'from': '/tf', 'to': '/vicon/tf'}
    assert includes['apriltag_localization.launch.xml'] == {
        'tag_pose_topic': '$(var vicon_pose_topic)',
        'world_frame': '$(var world_frame)',
        'detected_tag_frame': '$(var vicon_child_frame)',
        'tag_mount_frame': '$(var mount_frame)',
        'pelvis_frame': '$(var pelvis_frame)',
        'use_sim_time': '$(var use_sim_time)',
    }


def test_vicon_launch_anchors_fusion_world_on_one_camera():
    root = ET.parse(VICON_LAUNCH).getroot()
    args = _arguments(root)
    assert args['zed_anchor'] == 'true'
    assert args['zed_camera_pose_topic'] == '/vicon/zed_10025986/zed_10025986'
    assert args['zed_camera_frame'] == 'zed_right_left_camera_optical_frame'
    assert args['fusion_frame'] == 'fusion_world'
    mount = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
    for name in mount:
        assert args[f'zed_mount_{name}'] == '0.0'

    groups = [
        group for group in root.findall('group')
        if group.attrib.get('if') == '$(var zed_anchor)'
    ]
    assert len(groups) == 1
    nodes = {node.attrib['exec']: node for node in groups[0].findall('node')}
    assert set(nodes) == {'fusion_world_anchor_node', 'skeleton_viz_node'}
    assert nodes['skeleton_viz_node'].attrib['pkg'] == 'g1_human'
    assert _parameters(nodes['skeleton_viz_node']) == {
        'use_sim_time': '$(var use_sim_time)',
    }
    node = nodes['fusion_world_anchor_node']
    assert node.attrib == {
        'pkg': 'g1_navigation',
        'exec': 'fusion_world_anchor_node',
        'name': 'fusion_world_anchor_node',
        'output': 'screen',
    }
    expected = {
        'camera_pose_topic': '$(var zed_camera_pose_topic)',
        'world_frame': '$(var world_frame)',
        'fusion_frame': '$(var fusion_frame)',
        'camera_frame': '$(var zed_camera_frame)',
        'use_sim_time': '$(var use_sim_time)',
    }
    expected.update({
        f'mount_{name}': f'$(var zed_mount_{name})' for name in mount
    })
    assert _parameters(node) == expected


def test_anchor_executable_is_installable_and_thin():
    assert os.access(ANCHOR_NODE, os.X_OK)
    assert ANCHOR_NODE.read_text().splitlines()[-2:] == [
        "if __name__ == '__main__':",
        '    main()',
    ]
    cmake = CMAKE_PATH.read_text()
    assert 'scripts/fusion_world_anchor_node.py' in cmake
    assert 'RENAME fusion_world_anchor_node' in cmake


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

    # The simulator's clock and pelvis pose come from this launch, so it can
    # run before the bringup.
    relay = next(
        node for node in nodes if node.attrib['exec'] == 'sim_relay_node')
    assert relay.attrib['pkg'] == 'g1_bridge'
    relay_params = {
        param.attrib['name']: param.attrib['value']
        for param in relay.findall('param')
    }
    assert relay_params == {
        'network_interface': '$(var network_interface)',
        'domain_id': '$(var domain_id)',
        'use_sim_time': 'false',
    }
    args = {
        arg.attrib['name']: arg.attrib.get('default')
        for arg in root.findall('arg')
    }
    assert args['network_interface'] == 'lo'
    assert args['domain_id'] == '0'
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

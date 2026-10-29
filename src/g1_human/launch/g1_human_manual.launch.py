"""Simulated humans for the human-collision CBF, each with its own GUI.

Human i runs in namespace human_<i>: its human_control_gui publishes
/human_<i>/pose_command and /human_<i>/joint_commands, and its g1_human_node
publishes /human_<i>/colliders. collider_merge_node concatenates the fresh
arrays into /human/colliders, the topic the CBF nodes read. Every human also
draws on /human/collider_markers, in a marker namespace of its own.

    ros2 launch g1_human g1_human_manual.launch.py num_humans:=3
"""

import math
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# The /human/<topic> names g1_human_node and human_control_gui use, remapped
# into each human's namespace.
HUMAN_TOPICS = ('colliders', 'joint_commands', 'pose_command')


def human_layout(num_humans, x, y, yaw_deg, spacing):
    """Place the humans in a row along world y, centred on (x, y).

    Returns (namespace, x, y, yaw_deg) for each human.
    """
    if num_humans < 1:
        raise ValueError(f'num_humans must be at least 1, got {num_humans}')
    return [
        (f'human_{i}', x, y + (i - 0.5 * (num_humans - 1)) * spacing, yaw_deg)
        for i in range(num_humans)
    ]


def _launch_humans(context):
    def argument(name):
        return LaunchConfiguration(name).perform(context)

    use_sim_time = LaunchConfiguration('use_sim_time')
    # Read here instead of passed as a parameter file: its g1_human_node key
    # does not match the namespaced /human_<i>/g1_human_node.
    human_yaml = os.path.join(
        get_package_share_directory('g1_human'), 'config', 'human.yaml',
    )
    with open(human_yaml) as f:
        human_params = yaml.safe_load(f)['g1_human_node']['ros__parameters']

    layout = human_layout(
        int(argument('num_humans')),
        float(argument('human_x')),
        float(argument('human_y')),
        float(argument('human_yaw_deg')),
        float(argument('human_spacing')),
    )
    actions = []
    for namespace, x, y, yaw_deg in layout:
        remappings = [
            (f'/human/{topic}', f'/{namespace}/{topic}')
            for topic in HUMAN_TOPICS
        ]
        actions.append(Node(
            package='g1_human',
            executable='g1_human_node',
            name='g1_human_node',
            namespace=namespace,
            output='screen',
            parameters=[{
                **human_params,
                'human_x': x,
                'human_y': y,
                'human_yaw': math.radians(yaw_deg),
                'marker_namespace': f'{namespace}_colliders',
                'use_sim_time': use_sim_time,
            }],
            remappings=remappings,
        ))
        actions.append(Node(
            package='g1_human',
            executable='human_control_gui',
            name='human_control_gui',
            namespace=namespace,
            output='screen',
            parameters=[{
                'default_x': x,
                'default_y': y,
                'default_yaw_deg': yaw_deg,
                'publish_joints_default': True,
                'use_sim_time': use_sim_time,
            }],
            remappings=remappings,
        ))

    actions.append(Node(
        package='g1_human',
        executable='collider_merge_node',
        name='collider_merge_node',
        output='screen',
        parameters=[{
            'input_topics': [
                f'/{namespace}/colliders' for namespace, *_ in layout
            ],
            'output_topic': '/human/colliders',
            'use_sim_time': use_sim_time,
        }],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'num_humans', default_value='1',
            description='Number of simulated humans, each with its own '
                        'g1_human_node and control GUI',
        ),
        DeclareLaunchArgument(
            'human_x', default_value='0.5',
            description='World x of the row of humans [m]',
        ),
        DeclareLaunchArgument(
            'human_y', default_value='0.0',
            description='World y of the centre of the row [m]',
        ),
        DeclareLaunchArgument(
            'human_yaw_deg', default_value='0.0',
            description='Initial yaw of every human [deg]',
        ),
        DeclareLaunchArgument(
            'human_spacing', default_value='1.0',
            description='Distance between neighbouring humans along world y [m]',
        ),
        OpaqueFunction(function=_launch_humans),
    ])

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mpc',
            executable='mpc_node.py',
            name='mpc_node',
            remappings=[
                ("/ego_racecar/odom", "/pf/viz/inferred_pose")
            ],
        ),
    ])
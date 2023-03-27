import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share_directory = os.path.join(
        get_package_share_directory('mpc'),
        'waypoints', "")
    return LaunchDescription([
        Node(
            package='mpc',
            executable='mpc_node.py',
            name='mpc_node',
            parameters=[
                {
                'waypoints_path' : share_directory + "waypoints_sim.csv"
                }
            ],
            output='screen', # this is to see print statements
        ),
        Node(
            package='pure_pursuit',
            executable='pose_fake_pub_node',
            name='pose_fake_pub_node',
        )
    ])
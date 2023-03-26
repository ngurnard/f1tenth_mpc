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
            parameters=[
                {
                'waypoints_path' : "/home/nicholas/Documents/upenn/ESE615/sim_ws/src/lab-7-model-predictive-control-hot-wheels/mpc/waypoints/waypoints_mpc.csv"
                }
            ],
        ),
        Node(
            package='pure_pursuit',
            executable='pose_fake_pub_node',
            name='pose_fake_pub_node',
        )
    ])
"""
Full-system bringup: micro-ROS agent + UWB-to-odom + fusion filter +
NLOS context classification + live visualizer, in one launch.

Usage:
  ros2 launch uwb_position bringup_launch.py filter:=eskf   (default)
  ros2 launch uwb_position bringup_launch.py filter:=ekf
  ros2 launch uwb_position bringup_launch.py filter:=ukf

Start this BEFORE powering on / resetting the tag — the tag's micro-ROS
setup only attempts the agent handshake once at boot, with no retry loop,
so the agent must already be listening on UDP 8888 when the tag starts.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    fusion_launch_path = os.path.join(
        get_package_share_directory('uwb_position'), 'launch', 'fusion_launch.py'
    )

    micro_ros_agent = ExecuteProcess(
        cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'udp4', '--port', '8888'],
        output='screen'
    )

    fusion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(fusion_launch_path),
        launch_arguments={'filter': LaunchConfiguration('filter')}.items()
    )

    visualizer = Node(
        package='uwb_position',
        executable='fused_odom_node',
        name='fused_odom_viz',
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'filter',
            default_value='eskf',
            description='Filter type: ekf | ukf | eskf'
        ),
        micro_ros_agent,
        fusion,
        visualizer,
    ])

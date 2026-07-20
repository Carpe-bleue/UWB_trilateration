from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def _nodes(context, *args, **kwargs):
    config_dir = os.path.join(
        get_package_share_directory('uwb_position'), 'config'
    )
    filter_type = LaunchConfiguration('filter').perform(context)

    uwb_to_odom = Node(
        package='uwb_position',
        executable='uwb_to_odom',
        name='uwb_to_odom',
        output='screen'
    )

    if filter_type == 'eskf':
        # Custom ESKF node — IMU propagates nominal state at full rate;
        # UWB corrections update the error state only.
        # No robot_localization dependency required.
        fusion = Node(
            package='uwb_position',
            executable='eskf_node',
            name='eskf_node',
            output='screen'
        )
    elif filter_type == 'ukf':
        fusion = Node(
            package='robot_localization',
            executable='ukf_node',
            name='ukf_filter_node',
            output='screen',
            parameters=[os.path.join(config_dir, 'ukf.yaml')]
        )
    else:  # default: ekf
        fusion = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[os.path.join(config_dir, 'ekf.yaml')]
        )

    nlos_context = Node(
        package='uwb_position',
        executable='nlos_context_node',
        name='nlos_context_node',
        output='screen'
    )

    return [uwb_to_odom, fusion, nlos_context]


def generate_launch_description():
    return LaunchDescription([
        # ros2 launch uwb_position fusion_launch.py filter:=eskf
        # ros2 launch uwb_position fusion_launch.py filter:=ekf   (default)
        # ros2 launch uwb_position fusion_launch.py filter:=ukf
        DeclareLaunchArgument(
            'filter',
            default_value='ekf',
            description='Filter type: ekf | ukf | eskf'
        ),
        OpaqueFunction(function=_nodes),
    ])

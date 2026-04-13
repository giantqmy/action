import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('llama_behavior_ros2')
    param_file = os.path.join(pkg_dir, 'config', 'llama_params.yaml')

    llama_node = Node(
        package='llama_behavior_ros2',
        executable='llama_behavior_node',
        name='llama_behavior_node',
        output='screen',
        parameters=[param_file],
    )

    return LaunchDescription([llama_node])

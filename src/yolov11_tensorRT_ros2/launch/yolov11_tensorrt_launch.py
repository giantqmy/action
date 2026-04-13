import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('yolov11_tensorRT_ros2')
    param_file = os.path.join(pkg_dir, 'config', 'yolov11_params.yaml')

    yolov11_node = Node(
        package='yolov11_tensorRT_ros2',
        executable='yolov11_tensorrt_node',
        name='yolov11_tensorrt_node',
        output='screen',
        parameters=[param_file],
    )

    return LaunchDescription([yolov11_node])

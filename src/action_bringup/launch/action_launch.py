import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # YOLO detection node
    yolo_pkg = get_package_share_directory('yolov11_tensorRT_ros2')
    yolo_params = os.path.join(yolo_pkg, 'config', 'yolov11_params.yaml')

    yolo_node = Node(
        package='yolov11_tensorRT_ros2',
        executable='yolov11_tensorrt_node',
        name='yolov11_tensorrt_node',
        output='screen',
        parameters=[yolo_params],
    )

    # Llama behavior recognition node
    llama_pkg = get_package_share_directory('llama_behavior_ros2')
    llama_params = os.path.join(llama_pkg, 'config', 'llama_params.yaml')

    llama_node = Node(
        package='llama_behavior_ros2',
        executable='llama_behavior_node',
        name='llama_behavior_node',
        output='screen',
        parameters=[llama_params],
    )

    return LaunchDescription([yolo_node, llama_node])

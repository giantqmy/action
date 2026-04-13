#include "yolov11_tensorRT_ros2/yolov11_tensorrt_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    auto node = std::make_shared<yolov11_tensorRT_ros2::Yolov11TensorRTNode>(options);

    RCLCPP_INFO(node->get_logger(), "YOLOv11 TensorRT ROS2 node started, spinning...");
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}

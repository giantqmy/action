#ifndef YOLOV11_TENSORRT_ROS2_NODE_HPP
#define YOLOV11_TENSORRT_ROS2_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include "CNN.hpp"

#include <string>
#include <vector>
#include <memory>

namespace yolov11_tensorRT_ros2
{

class Yolov11TensorRTNode : public rclcpp::Node
{
public:
    explicit Yolov11TensorRTNode(const rclcpp::NodeOptions &options);
    ~Yolov11TensorRTNode();

private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);
    vision_msgs::msg::Detection2DArray buildDetectionMsg(
        const std::vector<float> &detections,
        int img_width, int img_height,
        const std_msgs::msg::Header &header);
    cv::Mat drawDetections(
        const cv::Mat &image,
        const std::vector<float> &detections,
        int img_width, int img_height);

    // Parameters
    std::string onnx_path_;
    std::string trt_path_;
    int input_width_;
    int input_height_;
    int input_channel_;
    int batch_size_;
    float object_thresh_;
    float nms_thresh_;
    int class_num_;
    bool publish_result_image_;
    std::vector<std::string> class_names_;

    // Inference engine
    std::unique_ptr<CNN> detector_;

    // ROS2 interfaces
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    image_transport::Publisher result_image_pub_;
};

} // namespace yolov11_tensorRT_ros2

#endif // YOLOV11_TENSORRT_ROS2_NODE_HPP

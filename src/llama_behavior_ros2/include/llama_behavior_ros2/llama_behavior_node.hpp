#ifndef LLAMA_BEHAVIOR_ROS2_NODE_HPP
#define LLAMA_BEHAVIOR_ROS2_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <future>
#include <opencv2/opencv.hpp>

namespace llama_behavior_ros2
{

using DetectionSync = message_filters::sync_policies::ApproximateTime<
    vision_msgs::msg::Detection2DArray,
    sensor_msgs::msg::Image>;

class LlamaBehaviorNode : public rclcpp::Node
{
public:
    explicit LlamaBehaviorNode(const rclcpp::NodeOptions &options);
    ~LlamaBehaviorNode();

private:
    void syncCallback(
        const vision_msgs::msg::Detection2DArray::ConstSharedPtr det_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr img_msg);

    std::string cropAndEncodeBase64(const cv::Mat &image, const vision_msgs::msg::Detection2D &det);
    std::string callLlamaServer(const std::vector<std::string> &images_b64,
                                 const std::vector<std::string> &class_names);

    // Parameters
    std::string server_url_;
    std::string prompt_;
    double confidence_threshold_;
    std::vector<std::string> target_classes_;  // empty = all classes
    int max_detections_;
    int queue_size_;
    int min_crop_size_;

    // ROS2 interfaces
    message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
    std::shared_ptr<message_filters::Synchronizer<DetectionSync>> sync_;
    rclcpp::Publisher<std_msgs::msg::String> behavior_pub_;

    // Async processing
    std::atomic<bool> processing_{false};
};

} // namespace llama_behavior_ros2

#endif // LLAMA_BEHAVIOR_ROS2_NODE_HPP

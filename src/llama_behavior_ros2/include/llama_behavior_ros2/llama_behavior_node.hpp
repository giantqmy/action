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
#include <image_transport/image_transport.hpp>

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

/// Behavior class definition (C++ equivalent of Python _default_classes)
struct BehaviorClass
{
    std::string id;         // e.g. "0"
    std::string label_cn;   // Chinese label
    std::string label_en;   // English label
    std::string severity;   // critical / warning / normal
    std::string description;
};

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

    /// Build the categories_text string from behavior_classes_
    std::string buildCategoriesText() const;

    /// Build the valid_ids string (comma-separated) from behavior_classes_
    std::string buildValidIds() const;

    /// Build the final prompt by filling placeholders in prompt_template_
    std::string buildPrompt() const;

    /// Default behavior class definitions (水域安全行为识别)
    static std::vector<BehaviorClass> defaultBehaviorClasses();

    /// Get severity color (BGR): critical=red, warning=yellow, normal=green
    static cv::Scalar severityColor(const std::string &severity);

    /// Draw behavior annotations on full image for rviz2 visualization
    cv::Mat drawBehaviorAnnotations(
        const cv::Mat &image,
        const vision_msgs::msg::Detection2DArray &dets,
        const std::vector<int> &valid_indices,
        const std::vector<std::string> &behavior_results);

    // Parameters
    std::string server_url_;
    std::string prompt_template_;  // prompt template with {categories_text} and {valid_ids} placeholders
    std::string prompt_;           // final assembled prompt
    double confidence_threshold_;
    std::vector<std::string> target_classes_;  // empty = all classes
    int max_detections_;
    int queue_size_;
    int min_crop_size_;
    bool publish_result_image_;

    // Behavior classes
    std::vector<BehaviorClass> behavior_classes_;

    // ROS2 interfaces
    message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
    std::shared_ptr<message_filters::Synchronizer<DetectionSync>> sync_;
    rclcpp::Publisher<std_msgs::msg::String> behavior_pub_;
    image_transport::Publisher result_image_pub_;

    // Async processing
    std::atomic<bool> processing_{false};
};

} // namespace llama_behavior_ros2

#endif // LLAMA_BEHAVIOR_ROS2_NODE_HPP

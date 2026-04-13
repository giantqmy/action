#include "yolov11_tensorRT_ros2/yolov11_tensorrt_node.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <functional>
#include <ament_index_cpp/get_package_share_directory.hpp>

namespace yolov11_tensorRT_ros2
{

Yolov11TensorRTNode::Yolov11TensorRTNode(const rclcpp::NodeOptions &options)
    : Node("yolov11_tensorrt_node", options)
{
    // ======================== declare and get parameters ========================
    declare_parameter<std::string>("onnx_path", "");
    declare_parameter<std::string>("trt_path", "");
    declare_parameter<int>("input_width", 640);
    declare_parameter<int>("input_height", 640);
    declare_parameter<int>("input_channel", 3);
    declare_parameter<int>("batch_size", 1);
    declare_parameter<float>("object_thresh", 0.5f);
    declare_parameter<float>("nms_thresh", 0.45f);
    declare_parameter<int>("class_num", 80);
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("detection_topic", "/yolov11/detections");
    declare_parameter<std::string>("result_image_topic", "/yolov11/result_image");
    declare_parameter<bool>("publish_result_image", true);
    declare_parameter<std::vector<std::string>>("class_names", std::vector<std::string>{});

    onnx_path_ = get_parameter("onnx_path").as_string();
    trt_path_ = get_parameter("trt_path").as_string();
    input_width_ = get_parameter("input_width").as_int();
    input_height_ = get_parameter("input_height").as_int();
    input_channel_ = get_parameter("input_channel").as_int();
    batch_size_ = get_parameter("batch_size").as_int();
    object_thresh_ = get_parameter("object_thresh").as_double();
    nms_thresh_ = get_parameter("nms_thresh").as_double();
    class_num_ = get_parameter("class_num").as_int();
    std::string image_topic = get_parameter("image_topic").as_string();
    std::string detection_topic = get_parameter("detection_topic").as_string();
    std::string result_image_topic = get_parameter("result_image_topic").as_string();
    publish_result_image_ = get_parameter("publish_result_image").as_bool();
    class_names_ = get_parameter("class_names").as_string_array();

    // Resolve relative paths against package share directory
    auto resolvePath = [&](const std::string &path) -> std::string {
        if (path.empty() || path[0] == '/') return path;
        // If path starts with "install/", resolve from package share
        if (path.find("install/") == 0) {
            std::string pkg_share = ament_index_cpp::get_package_share_directory("yolov11_tensorRT_ros2");
            // strip "install/yolov11_tensorRT_ros2/share/yolov11_tensorRT_ros2/" prefix
            auto pos = path.find("share/yolov11_tensorRT_ros2/");
            if (pos != std::string::npos) {
                return pkg_share + "/" + path.substr(pos + std::string("share/yolov11_tensorRT_ros2/").size());
            }
            return path;
        }
        return path;
    };

    onnx_path_ = resolvePath(onnx_path_);
    trt_path_ = resolvePath(trt_path_);

    RCLCPP_INFO(get_logger(), "ONNX path: %s", onnx_path_.c_str());
    RCLCPP_INFO(get_logger(), "TRT path:  %s", trt_path_.c_str());
    RCLCPP_INFO(get_logger(), "Input size: %dx%dx%d", input_width_, input_height_, input_channel_);

    // ======================== init detector ========================
    detector_ = std::make_unique<CNN>(onnx_path_, trt_path_, batch_size_, input_channel_, input_width_, input_height_);
    RCLCPP_INFO(get_logger(), "YOLOv11 TensorRT detector initialized.");

    // ======================== create subscribers and publishers ========================
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10,
        std::bind(&Yolov11TensorRTNode::imageCallback, this, std::placeholders::_1));

    detection_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(detection_topic, 10);

    if (publish_result_image_) {
        result_image_pub_ = image_transport::create_publisher(this, result_image_topic);
    }

    RCLCPP_INFO(get_logger(), "Subscribed to: %s", image_topic.c_str());
    RCLCPP_INFO(get_logger(), "Publishing detections to: %s", detection_topic.c_str());
    if (publish_result_image_) {
        RCLCPP_INFO(get_logger(), "Publishing result image to: %s", result_image_topic.c_str());
    }
}

Yolov11TensorRTNode::~Yolov11TensorRTNode()
{
    RCLCPP_INFO(get_logger(), "YOLOv11 TensorRT node shutting down.");
}

void Yolov11TensorRTNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    // ============================================================
    // 时延测量: 各阶段打点，通过 RCLCPP_INFO 输出毫秒级耗时
    // 优化时重点关注 t_infer（TensorRT推理）和 t_total（端到端）
    // ============================================================
    auto t_cb_start = std::chrono::high_resolution_clock::now();

    // [阶段1] Convert ROS image to OpenCV
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    auto t_convert = std::chrono::high_resolution_clock::now();

    cv::Mat image = cv_ptr->image;
    int img_width = image.cols;
    int img_height = image.rows;

    // [阶段2] TensorRT 推理 — 这是最大的时延瓶颈，优化时优先关注
    detector_->Inference(image);
    auto t_infer = std::chrono::high_resolution_clock::now();

    int num_objects = static_cast<int>(detector_->DetectiontRects_.size()) / 6;

    // [阶段3] 构建检测结果消息
    auto det_msg = buildDetectionMsg(detector_->DetectiontRects_, img_width, img_height, msg->header);
    auto t_build_msg = std::chrono::high_resolution_clock::now();

    // [阶段4] 绘制检测框并发布可视化图像
    if (publish_result_image_ && num_objects > 0) {
        cv::Mat result_image = drawDetections(image, detector_->DetectiontRects_, img_width, img_height);
        auto result_msg = cv_bridge::CvImage(msg->header, "bgr8", result_image).toImageMsg();
        result_image_pub_.publish(result_msg);
    }
    auto t_draw = std::chrono::high_resolution_clock::now();

    // [阶段5] 发布检测结果
    detection_pub_->publish(det_msg);
    auto t_publish = std::chrono::high_resolution_clock::now();

    // ---- 输出各阶段耗时 ----
    float ms_convert   = std::chrono::duration<float, std::milli>(t_convert - t_cb_start).count();
    float ms_infer     = std::chrono::duration<float, std::milli>(t_infer - t_convert).count();
    float ms_build_msg = std::chrono::duration<float, std::milli>(t_build_msg - t_infer).count();
    float ms_draw      = std::chrono::duration<float, std::milli>(t_draw - t_build_msg).count();
    float ms_publish   = std::chrono::duration<float, std::milli>(t_publish - t_draw).count();
    float ms_total     = std::chrono::duration<float, std::milli>(t_publish - t_cb_start).count();

    RCLCPP_INFO(get_logger(),
        "[YOLO-TIMING] convert=%.1fms | infer=%.1fms | build_msg=%.1fms | draw=%.1fms | "
        "publish=%.1fms | total=%.1fms | objects=%d",
        ms_convert, ms_infer, ms_build_msg, ms_draw, ms_publish, ms_total, num_objects);
}

vision_msgs::msg::Detection2DArray Yolov11TensorRTNode::buildDetectionMsg(
    const std::vector<float> &detections,
    int img_width, int img_height,
    const std_msgs::msg::Header &header)
{
    vision_msgs::msg::Detection2DArray det_array;
    det_array.header = header;

    for (size_t i = 0; i + 5 < detections.size(); i += 6) {
        int class_id = static_cast<int>(detections[i + 0]);
        float conf = detections[i + 1];
        float xmin = detections[i + 2];
        float ymin = detections[i + 3];
        float xmax = detections[i + 4];
        float ymax = detections[i + 5];

        vision_msgs::msg::Detection2D det;
        det.header = header;

        // Bounding box in pixel coordinates
        float cx = (xmin + xmax) / 2.0f * img_width;
        float cy = (ymin + ymax) / 2.0f * img_height;
        float w  = (xmax - xmin) * img_width;
        float h  = (ymax - ymin) * img_height;

        det.bbox.center.position.x = cx;
        det.bbox.center.position.y = cy;
        det.bbox.size_x = w;
        det.bbox.size_y = h;

        // Result: class id + score
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = std::to_string(class_id);
        if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
            hyp.hypothesis.class_id = class_names_[class_id];
        }
        hyp.hypothesis.score = conf;
        det.results.push_back(hyp);

        det_array.detections.push_back(det);
    }

    return det_array;
}

cv::Mat Yolov11TensorRTNode::drawDetections(
    const cv::Mat &image,
    const std::vector<float> &detections,
    int img_width, int img_height)
{
    cv::Mat result = image.clone();

    for (size_t i = 0; i + 5 < detections.size(); i += 6) {
        int class_id = static_cast<int>(detections[i + 0]);
        float conf = detections[i + 1];
        int xmin = static_cast<int>(detections[i + 2] * img_width + 0.5f);
        int ymin = static_cast<int>(detections[i + 3] * img_height + 0.5f);
        int xmax = static_cast<int>(detections[i + 4] * img_width + 0.5f);
        int ymax = static_cast<int>(detections[i + 5] * img_height + 0.5f);

        cv::rectangle(result, cv::Point(xmin, ymin), cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), 2);

        std::string label;
        if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
            label = class_names_[class_id] + ": ";
        }
        char score_str[32];
        snprintf(score_str, sizeof(score_str), "%.2f", conf);
        label += score_str;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(result,
                      cv::Point(xmin, ymin - text_size.height - 4),
                      cv::Point(xmin + text_size.width, ymin),
                      cv::Scalar(0, 255, 0), -1);
        cv::putText(result, label, cv::Point(xmin, ymin - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return result;
}

} // namespace yolov11_tensorRT_ros2

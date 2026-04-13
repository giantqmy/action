#include "llama_behavior_ros2/llama_behavior_node.hpp"

#include <curl/curl.h>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace llama_behavior_ros2
{

// ======================== Base64 encoding ========================
static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64Encode(const unsigned char *data, size_t len)
{
    std::string result;
    result.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned int n = (static_cast<unsigned int>(data[i]) << 16);
        if (i + 1 < len) n |= (static_cast<unsigned int>(data[i + 1]) << 8);
        if (i + 2 < len) n |= static_cast<unsigned int>(data[i + 2]);

        result += base64_chars[(n >> 18) & 0x3F];
        result += base64_chars[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? base64_chars[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? base64_chars[n & 0x3F] : '=';
    }
    return result;
}

// ======================== JSON string escape ========================
static std::string jsonEscape(const std::string &s)
{
    std::string result;
    result.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n";  break;
            case '\r': result += "\\r";  break;
            case '\t': result += "\\t";  break;
            default:   result += c;      break;
        }
    }
    return result;
}

// ======================== CURL write callback ========================
static size_t curlWriteCallback(void *contents, size_t size, size_t nmemb, std::string *output)
{
    size_t total = size * nmemb;
    output->append(static_cast<char *>(contents), total);
    return total;
}

// ======================== Default behavior classes ========================
std::vector<BehaviorClass> LlamaBehaviorNode::defaultBehaviorClasses()
{
    return {
        {"0", "溺水",     "drowning",      "critical", "四肢无规律挣扎，有溺水风险。"},
        {"1", "游泳",     "swimming",      "normal",   "人员在水中正常游泳。"},
        {"2", "攀爬栏杆", "climbing",      "warning",  "人员攀爬或翻越栏杆。"},
        {"3", "正常行走", "normal_walking", "normal",   "岸上人员正常行走或站立。"},
        {"4", "正在救援", "waterhelping",  "normal",   "水中人员抱住红色救生圈。"},
        {"5", "在船上",   "aboard",        "normal",   "人员在船上或在开船。"},
    };
}

// ======================== Build categories_text ========================
std::string LlamaBehaviorNode::buildCategoriesText() const
{
    // Format: "0: 溺水 (drowning, critical) - 四肢无规律挣扎，有溺水风险。\n..."
    std::ostringstream ss;
    for (size_t i = 0; i < behavior_classes_.size(); ++i) {
        const auto &bc = behavior_classes_[i];
        ss << bc.id << ": " << bc.label_cn << " (" << bc.label_en << ", " << bc.severity
           << ") - " << bc.description;
        if (i + 1 < behavior_classes_.size()) ss << "\n";
    }
    return ss.str();
}

// ======================== Build valid_ids ========================
std::string LlamaBehaviorNode::buildValidIds() const
{
    std::ostringstream ss;
    for (size_t i = 0; i < behavior_classes_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << behavior_classes_[i].id;
    }
    return ss.str();
}

// ======================== Build prompt ========================
std::string LlamaBehaviorNode::buildPrompt() const
{
    std::string prompt = prompt_template_;

    const std::string placeholder_categories = "{categories_text}";
    const std::string placeholder_ids = "{valid_ids}";

    std::string categories_text = buildCategoriesText();
    std::string valid_ids = buildValidIds();

    // Replace {categories_text}
    size_t pos = 0;
    while ((pos = prompt.find(placeholder_categories, pos)) != std::string::npos) {
        prompt.replace(pos, placeholder_categories.length(), categories_text);
        pos += categories_text.length();
    }

    // Replace {valid_ids}
    pos = 0;
    while ((pos = prompt.find(placeholder_ids, pos)) != std::string::npos) {
        prompt.replace(pos, placeholder_ids.length(), valid_ids);
        pos += valid_ids.length();
    }

    return prompt;
}

// ======================== Constructor ========================
LlamaBehaviorNode::LlamaBehaviorNode(const rclcpp::NodeOptions &options)
    : Node("llama_behavior_node", options)
{
    // Default prompt template (water safety behavior recognition)
    // Note: {categories_text} and {valid_ids} are placeholders filled at runtime.
    // Single braces in the JSON example are literal for the model, NOT placeholders.
    const std::string default_prompt_template =
        "你是一个智能水域安全行为识别系统，部署于监控摄像头端，负责实时分析画面中人物的行为并判断安全风险等级。\n"
        "\n"
        "## 识别原则\n"
        "仅根据画面中可见的动作姿态进行判断，不做臆测或推断\n"
        "\n"
        "## 可识别的行为类别\n"
        "{categories_text}\n"
        "\n"
        "## 输出要求\n"
        "请严格按以下 JSON 格式输出，不要包含其他内容：\n"
        "```json\n"
        "{\n"
        "  \"behavior_id\": \"<行为ID>\",\n"
        "  \"behavior_label\": \"<行为英文标签>\",\n"
        "  \"description\": \"<简练行为描述>\",\n"
        "  \"severity\": \"<严重等级: critical/warning/normal>\",\n"
        "  \"confidence\": <0.7-1.0的置信度>\n"
        "}\n"
        "```\n"
        "\n"
        "behavior_id 必须是以下之一: {valid_ids}\n"
        "如果无法确定行为，返回 unknown。\n"
        "请基于图像内容客观分析，不要臆测。";

    // Declare and get parameters
    declare_parameter<std::string>("server_url", "http://127.0.0.1:8080/v1/chat/completions");
    declare_parameter<std::string>("prompt_template", default_prompt_template);
    declare_parameter<double>("confidence_threshold", 0.5);
    declare_parameter<std::vector<std::string>>("target_classes", std::vector<std::string>{"person"});
    declare_parameter<int>("max_detections", 3);
    declare_parameter<int>("queue_size", 10);
    declare_parameter<std::string>("detection_topic", "/yolov11/detections");
    declare_parameter<std::string>("image_topic", "/camera/image_raw");
    declare_parameter<std::string>("behavior_topic", "/llama/behavior");
    declare_parameter<int>("min_crop_size", 50);

    server_url_ = get_parameter("server_url").as_string();
    prompt_template_ = get_parameter("prompt_template").as_string();
    confidence_threshold_ = get_parameter("confidence_threshold").as_double();
    target_classes_ = get_parameter("target_classes").as_string_array();
    max_detections_ = get_parameter("max_detections").as_int();
    queue_size_ = get_parameter("queue_size").as_int();
    min_crop_size_ = get_parameter("min_crop_size").as_int();

    // Initialize behavior classes
    behavior_classes_ = defaultBehaviorClasses();

    // Build the final prompt by filling placeholders
    prompt_ = buildPrompt();

    std::string detection_topic = get_parameter("detection_topic").as_string();
    std::string image_topic = get_parameter("image_topic").as_string();
    std::string behavior_topic = get_parameter("behavior_topic").as_string();

    RCLCPP_INFO(get_logger(), "Server URL: %s", server_url_.c_str());
    RCLCPP_INFO(get_logger(), "Behavior classes: %zu loaded", behavior_classes_.size());
    RCLCPP_INFO(get_logger(), "Confidence threshold: %.2f", confidence_threshold_);
    RCLCPP_INFO(get_logger(), "Min crop size: %d px", min_crop_size_);

    if (!target_classes_.empty()) {
        std::string classes_str;
        for (size_t i = 0; i < target_classes_.size(); ++i) {
            if (i > 0) classes_str += ", ";
            classes_str += target_classes_[i];
        }
        RCLCPP_INFO(get_logger(), "Target classes: [%s]", classes_str.c_str());
    } else {
        RCLCPP_INFO(get_logger(), "Target classes: ALL");
    }

    RCLCPP_INFO(get_logger(), "Final prompt preview (first 120 chars): %.120s...", prompt_.c_str());

    // Initialize CURL globally
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Create synchronized subscribers (ApproximateTime sync)
    det_sub_.subscribe(this, detection_topic);
    img_sub_.subscribe(this, image_topic);

    sync_ = std::make_shared<message_filters::Synchronizer<DetectionSync>>(
        DetectionSync(queue_size_), det_sub_, img_sub_);
    sync_->registerCallback(&LlamaBehaviorNode::syncCallback, this);

    // Publisher
    behavior_pub_ = create_publisher<std_msgs::msg::String>(behavior_topic, 10);

    RCLCPP_INFO(get_logger(), "Llama behavior node initialized.");
    RCLCPP_INFO(get_logger(), "Subscribing: %s + %s", detection_topic.c_str(), image_topic.c_str());
    RCLCPP_INFO(get_logger(), "Publishing to: %s", behavior_topic.c_str());
}

LlamaBehaviorNode::~LlamaBehaviorNode()
{
    curl_global_cleanup();
}

// ======================== Sync callback ========================
void LlamaBehaviorNode::syncCallback(
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr det_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
    // ============================================================
    // 时延测量: LLaMA 节点的主流程各阶段打点
    // 其中 t_crop 到 t_thread_spawn 之间是同步预处理耗时
    // t_llama 是异步线程中的 HTTP 调用耗时（最大瓶颈）
    // ============================================================
    auto t_cb_start = std::chrono::high_resolution_clock::now();

    // Skip if still processing previous frame
    if (processing_.load()) {
        RCLCPP_DEBUG(get_logger(), "Skipping frame, still processing...");
        return;
    }

    // [阶段1] Convert ROS image to OpenCV
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    auto t_convert = std::chrono::high_resolution_clock::now();

    cv::Mat image = cv_ptr->image;

    // [阶段2] 过滤检测结果（按置信度 + 类别）
    std::vector<std::pair<int, std::string>> valid_dets;
    for (size_t i = 0; i < det_msg->detections.size(); ++i) {
        const auto &det = det_msg->detections[i];
        if (det.results.empty()) continue;

        float conf = det.results[0].hypothesis.score;
        std::string cls = det.results[0].hypothesis.class_id;

        if (conf < confidence_threshold_) continue;

        // Filter by target classes
        if (!target_classes_.empty()) {
            bool match = false;
            for (const auto &tc : target_classes_) {
                if (cls == tc) { match = true; break; }
            }
            if (!match) continue;
        }

        valid_dets.push_back({static_cast<int>(i), cls});
        if (static_cast<int>(valid_dets.size()) >= max_detections_) break;
    }
    auto t_filter = std::chrono::high_resolution_clock::now();

    if (valid_dets.empty()) return;

    // [阶段3] 裁剪目标区域 + JPEG编码 + Base64编码
    std::vector<std::string> images_b64;
    std::vector<std::string> class_names;

    for (const auto &[idx, cls] : valid_dets) {
        std::string b64 = cropAndEncodeBase64(image, det_msg->detections[idx]);
        if (!b64.empty()) {
            images_b64.push_back(b64);
            class_names.push_back(cls);
        }
    }
    auto t_crop = std::chrono::high_resolution_clock::now();

    if (images_b64.empty()) return;

    // 计算同步阶段耗时（这些在主线程阻塞）
    float ms_sync_convert = std::chrono::duration<float, std::milli>(t_convert - t_cb_start).count();
    float ms_sync_filter = std::chrono::duration<float, std::milli>(t_filter - t_convert).count();
    float ms_sync_crop   = std::chrono::duration<float, std::milli>(t_crop - t_filter).count();
    float ms_sync_total  = std::chrono::duration<float, std::milli>(t_crop - t_cb_start).count();

    RCLCPP_INFO(get_logger(),
        "[LLAMA-TIMING-SYNC] convert=%.1fms | filter=%.1fms | crop+encode=%.1fms | sync_total=%.1fms | dets=%zu",
        ms_sync_convert, ms_sync_filter, ms_sync_crop, ms_sync_total, images_b64.size());

    // [阶段4] 异步调用 LLaMA 服务器（HTTP请求是最大时延瓶颈）
    processing_.store(true);
    auto self = this->shared_from_this();
    std::thread([this, self, images_b64, class_names, t_crop]() {
        auto t_thread_start = std::chrono::high_resolution_clock::now();

        std::string result = callLlamaServer(images_b64, class_names);

        auto t_llama = std::chrono::high_resolution_clock::now();

        processing_.store(false);

        if (!result.empty()) {
            auto msg = std_msgs::msg::String();
            msg.data = result;
            behavior_pub_->publish(msg);
        }
        auto t_pub = std::chrono::high_resolution_clock::now();

        // 异步阶段耗时
        float ms_queue     = std::chrono::duration<float, std::milli>(t_thread_start - t_crop).count();
        float ms_llama     = std::chrono::duration<float, std::milli>(t_llama - t_thread_start).count();
        float ms_publish   = std::chrono::duration<float, std::milli>(t_pub - t_llama).count();
        float ms_async     = std::chrono::duration<float, std::milli>(t_pub - t_crop).count();

        RCLCPP_INFO(get_logger(),
            "[LLAMA-TIMING-ASYNC] queue=%.1fms | llama_http=%.1fms | publish=%.1fms | async_total=%.1fms",
            ms_queue, ms_llama, ms_publish, ms_async);

        if (!result.empty()) {
            RCLCPP_INFO(get_logger(), "Behavior result: %s", result.c_str());
        }
    }).detach();
}

// ======================== Crop and encode ========================
std::string LlamaBehaviorNode::cropAndEncodeBase64(
    const cv::Mat &image,
    const vision_msgs::msg::Detection2D &det)
{
    float cx = det.bbox.center.position.x;
    float cy = det.bbox.center.position.y;
    float w = det.bbox.size_x;
    float h = det.bbox.size_y;

    int x1 = std::max(0, static_cast<int>(cx - w / 2.0f));
    int y1 = std::max(0, static_cast<int>(cy - h / 2.0f));
    int x2 = std::min(image.cols, static_cast<int>(cx + w / 2.0f));
    int y2 = std::min(image.rows, static_cast<int>(cy + h / 2.0f));

    // Skip too small crops
    if ((x2 - x1) < min_crop_size_ || (y2 - y1) < min_crop_size_) {
        return "";
    }

    cv::Mat cropped = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

    // Encode as JPEG
    std::vector<uchar> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 85};
    cv::imencode(".jpg", cropped, buf, params);

    return base64Encode(buf.data(), buf.size());
}

// ======================== Call llama.cpp server ========================
std::string LlamaBehaviorNode::callLlamaServer(
    const std::vector<std::string> &images_b64,
    const std::vector<std::string> &class_names)
{
    auto t_fn_start = std::chrono::high_resolution_clock::now();

    // [细粒度1] 构建 JSON 请求体
    std::ostringstream json;
    json << "{"
         << "\"model\":\"ggml-model\","
         << "\"messages\":[{\"role\":\"user\",\"content\":[";

    // Add text prompt
    json << "{\"type\":\"text\",\"text\":\"" << jsonEscape(prompt_) << "\"},";

    // Add images
    for (size_t i = 0; i < images_b64.size(); ++i) {
        json << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,"
             << images_b64[i] << "\"}}";
        if (i + 1 < images_b64.size()) json << ",";
    }

    json << "]}],"
         << "\"max_tokens\":512,"
         << "\"temperature\":0.1"
         << "}";

    std::string json_str = json.str();
    auto t_json = std::chrono::high_resolution_clock::now();

    // [细粒度2] CURL 初始化
    CURL *curl = curl_easy_init();
    if (!curl) {
        RCLCPP_ERROR(get_logger(), "Failed to init CURL");
        return "";
    }

    std::string response;
    struct curl_slist *headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, server_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_str.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    auto t_curl_setup = std::chrono::high_resolution_clock::now();

    // [细粒度3] HTTP 请求 — 通常是最大时延瓶颈
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    auto t_http = std::chrono::high_resolution_clock::now();

    if (res != CURLE_OK) {
        RCLCPP_ERROR(get_logger(), "CURL error: %s", curl_easy_strerror(res));
        float ms_total = std::chrono::duration<float, std::milli>(t_http - t_fn_start).count();
        RCLCPP_WARN(get_logger(), "[LLAMA-TIMING-HTTP] failed after %.1fms (CURL error: %s)",
                     ms_total, curl_easy_strerror(res));
        return "";
    }

    // [细粒度4] 解析响应
    size_t content_pos = response.find("\"content\":\"");
    if (content_pos == std::string::npos) {
        content_pos = response.find("\"content\":");
        if (content_pos == std::string::npos) {
            RCLCPP_WARN(get_logger(), "Could not parse response: %s", response.c_str());
            return response.substr(0, 200);
        }
    }

    size_t start = response.find("\"", content_pos + 10);
    if (start == std::string::npos) return "";
    start++;

    std::string content;
    for (size_t i = start; i < response.size(); ++i) {
        if (response[i] == '\\' && i + 1 < response.size()) {
            switch (response[i + 1]) {
                case '"':  content += '"';  break;
                case 'n':  content += '\n'; break;
                case 't':  content += '\t'; break;
                case '\\': content += '\\'; break;
                default:   content += response[i + 1]; break;
            }
            ++i;
        } else if (response[i] == '"') {
            break;
        } else {
            content += response[i];
        }
    }
    auto t_parse = std::chrono::high_resolution_clock::now();

    // ---- 细粒度耗时输出 ----
    float ms_json_build  = std::chrono::duration<float, std::milli>(t_json - t_fn_start).count();
    float ms_curl_setup  = std::chrono::duration<float, std::milli>(t_curl_setup - t_json).count();
    float ms_http        = std::chrono::duration<float, std::milli>(t_http - t_curl_setup).count();
    float ms_parse       = std::chrono::duration<float, std::milli>(t_parse - t_http).count();
    float ms_total       = std::chrono::duration<float, std::milli>(t_parse - t_fn_start).count();

    RCLCPP_INFO(get_logger(),
        "[LLAMA-TIMING-HTTP] json_build=%.1fms | curl_setup=%.1fms | http_request=%.1fms | "
        "parse=%.1fms | total=%.1fms | response_size=%zu bytes",
        ms_json_build, ms_curl_setup, ms_http, ms_parse, ms_total, response.size());

    return content;
}

} // namespace llama_behavior_ros2

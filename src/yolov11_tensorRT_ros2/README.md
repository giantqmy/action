# yolov11_tensorRT_ros2

将 [yolov11_tensorRT_postprocess_cuda](https://github.com/cqu20160901/yolov11_tensorRT_postprocess_cuda) 改造为 ROS2 节点。

基于 TensorRT + CUDA 加速的 YOLOv11 目标检测 ROS2 功能包，支持订阅相机图像并实时发布检测结果。

## 功能特性

- **TensorRT FP16 推理** — 自动将 ONNX 模型转换为 TRT 引擎
- **CUDA 预处理 + 后处理** — GPU 加速图像缩放、归一化、NMS 前筛选
- **标准 ROS2 接口** — 订阅 `sensor_msgs/Image`，发布 `vision_msgs/Detection2DArray`
- **可视化输出** — 可选发布带检测框的标注图像
- **参数化配置** — 所有路径、阈值、话题名通过 YAML 参数文件配置

## 依赖

| 依赖 | 版本要求 |
|------|---------|
| ROS2 | Humble / Iron / Jazzy |
| CUDA | >= 11.x |
| TensorRT | >= 8.x (测试 8.6.1.6) |
| OpenCV | >= 4.x |
| cv_bridge | ROS2 对应版本 |
| image_transport | ROS2 对应版本 |
| vision_msgs | ROS2 对应版本 |

## 编译

```bash
# 1. 确保 ROS2 环境已 source
source /opt/ros/humble/setup.bash

# 2. 修改 CMakeLists.txt 中的 TensorRT 路径（如果不在 /usr/local/TensorRT）
#    set(TENSORRT_ROOT /your/tensorrt/path)

# 3. 编译
cd your_ws
colcon build --packages-select yolov11_tensorRT_ros2

# 4. source 工作空间
source install/setup.bash
```

## 运行

```bash
# 使用 launch 文件（推荐）
ros2 launch yolov11_tensorRT_ros2 yolov11_tensorrt_launch.py

# 或直接运行节点
ros2 run yolov11_tensorRT_ros2 yolov11_tensorrt_node --ros-args \
  --params-file install/yolov11_tensorRT_ros2/share/yolov11_tensorRT_ros2/config/yolov11_params.yaml
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `onnx_path` | — | ONNX 模型路径 |
| `trt_path` | — | TRT 引擎路径（不存在时自动从 ONNX 生成） |
| `input_width` | 640 | 模型输入宽度 |
| `input_height` | 640 | 模型输入高度 |
| `object_thresh` | 0.5 | 目标置信度阈值 |
| `nms_thresh` | 0.45 | NMS IOU 阈值 |
| `image_topic` | `/camera/image_raw` | 输入图像话题 |
| `detection_topic` | `/yolov11/detections` | 检测结果输出话题 |
| `result_image_topic` | `/yolov11/result_image` | 标注图像输出话题 |
| `publish_result_image` | true | 是否发布标注图像 |
| `class_names` | COCO 80 类 | 类别名称列表 |

## 话题

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/camera/image_raw` | `sensor_msgs/Image` | 订阅 | 输入 RGB 图像 |
| `/yolov11/detections` | `vision_msgs/Detection2DArray` | 发布 | 检测结果 |
| `/yolov11/result_image` | `sensor_msgs/Image` | 发布 | 带标注图像（可选） |

## 项目结构

```
yolov11_tensorRT_ros2/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── yolov11_params.yaml          # 参数配置
├── launch/
│   └── yolov11_tensorrt_launch.py   # launch 文件
├── include/yolov11_tensorRT_ros2/
│   ├── yolov11_tensorrt_node.hpp    # ROS2 节点头文件
│   ├── CNN.hpp                      # 推理引擎头文件
│   ├── postprocess_cuda.hpp         # 后处理头文件
│   └── common_struct.hpp            # 数据结构
├── src/
│   ├── main.cpp                     # 节点入口
│   ├── yolov11_tensorrt_node.cpp    # ROS2 节点实现
│   ├── CNN.cpp                      # TensorRT 推理实现
│   ├── postprocess_cuda.cpp         # NMS 后处理
│   ├── common/
│   │   ├── common.hpp               # ONNX→TRT 工具函数
│   │   └── logging.h                # TensorRT 日志
│   └── kernels/
│       ├── image_preprocess.cu/cuh  # CUDA 图像预处理
│       └── get_nms_before_boxes.cu/cuh  # CUDA NMS 前筛选
├── models/                          # 模型文件 (.onnx / .trt)
└── images/                          # 测试图片
```

## 使用示例

```bash
# 发布测试图像（使用 image_publisher）
ros2 run image_tools cam2image --ros-args -p frequency:=10.0

# 查看检测结果
ros2 topic echo /yolov11/detections

# 查看标注图像（使用 rqt_image_view）
ros2 run rqt_image_view rqt_image_view --ros-args -r /image:=/yolov11/result_image
```

## 注意事项

- 首次运行时会自动将 ONNX 模型转换为 TRT 引擎，耗时约 1-5 分钟
- TRT 引擎文件生成后会缓存，后续启动直接加载
- `GpuSrcImage_` 预分配了 810×1080×3 的显存，如需更大分辨率请修改 `CNN.cpp`
- CUDA 架构默认 `sm_89`（RTX 4090），其他 GPU 需修改 `CMakeLists.txt` 中的 `CUDA_GEN_CODE`

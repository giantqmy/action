# Action — YOLOv11 + LLaMA 行为识别 ROS2 工作空间

基于 **YOLOv11 TensorRT** 实时目标检测 + **LLaMA 多模态大模型** 行为理解的 ROS2 感知管线。

## 系统架构

```
Camera (/camera/image_raw)
    │
    ├──► YOLOv11 TensorRT 节点
    │      ├── 发布: /yolov11/detections  (Detection2DArray)
    │      └── 发布: /yolov11/result_image (Image, 带检测框)
    │
    └──► LLaMA Behavior 节点
           ├── 订阅: /camera/image_raw + /yolov11/detections
           ├── 同步 (ApproximateTime) → 裁剪目标区域 → Base64 编码
           ├── 调用 llama.cpp 多模态服务器 (HTTP OpenAI API)
           └── 发布: /llama/behavior (String)
```

## 工作空间结构

```
action/
└── src/
    ├── action_bringup/              # 联合启动包 (launch-only)
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   └── launch/
    │       └── action_launch.py
    │
    ├── yolov11_tensorRT_ros2/       # YOLOv11 TensorRT 检测节点
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── config/yolov11_params.yaml
    │   ├── launch/
    │   ├── models/                  # ONNX + TRT 模型文件
    │   ├── include/
    │   └── src/                     # C++ + CUDA kernel
    │
    └── llama_behavior_ros2/         # LLaMA 行为识别节点
        ├── CMakeLists.txt
        ├── package.xml
        ├── config/llama_params.yaml
        ├── launch/
        ├── include/
        └── src/
```

## 环境依赖

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| ROS2 | Humble / Iron | `ros-humble-desktop` 或 `ros-iron-desktop` |
| CUDA | 11.8+ | 需匹配你的 GPU 架构 |
| TensorRT | 8.5+ | YOLOv11 推理加速 |
| OpenCV | 4.x | 图像处理 |
| libcurl | - | HTTP 调用 llama.cpp 服务器 |
| llama.cpp | 最新 | 多模态推理服务端 |

### GPU 架构配置

在 `src/yolov11_tensorRT_ros2/CMakeLists.txt` 中，修改 CUDA 架构为你自己的 GPU：

```cmake
set(CUDA_GEN_CODE "-gencode=arch=compute_XX,code=sm_XX")
```

| GPU 系列 | compute / sm |
|-----------|-------------|
| RTX 20 系列 (Turing) | 75 |
| RTX 30 系列 (Ampere) | 80 / 86 |
| RTX 40 系列 (Ada) | 89 |
| Jetson Orin | 87 |

### TensorRT 路径

默认查找 `/usr/local/TensorRT`，如果你的 TensorRT 装在其他位置，编译时指定：

```bash
colcon build --cmake-args -DTENSORRT_ROOT=/path/to/your/TensorRT
```

## 快速开始

### 1. 启动 llama.cpp 多模态服务器

在另一个终端启动 llama.cpp 的 OpenAI 兼容服务器：

```bash
# 示例（根据你的模型路径调整）
./llama-server \
  -m your-multimodal-model.gguf \
  --mmproj your-mmproj.gguf \
  --host 0.0.0.0 \
  --port 8080
```

验证服务可用：
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4-vision-preview","messages":[{"role":"user","content":"hello"}]}'
```

### 2. 克隆 & 编译工作空间

```bash
# 克隆
mkdir -p ~/action_ws/src
cd ~/action_ws/src
git clone https://github.com/giantqmy/action.git .

# 回到工作空间根目录
cd ~/action_ws

# 编译（默认 TensorRT 路径）
colcon build --symlink-install

# 如果 TensorRT 不在默认路径
colcon build --symlink-install --cmake-args -DTENSORRT_ROOT=/your/tensorrt/path

# 加载环境
source install/setup.bash
```

### 3. 启动系统

**方式一：联合启动（推荐）**

```bash
source install/setup.bash
ros2 launch action_bringup action_launch.py
```

这会同时启动 YOLOv11 检测节点和 LLaMA 行为识别节点。

**方式二：单独启动**

终端 1 — YOLOv11 检测：
```bash
source install/setup.bash
ros2 launch yolov11_tensorRT_ros2 yolov11_tensorrt_launch.py
```

终端 2 — LLaMA 行为识别：
```bash
source install/setup.bash
ros2 launch llama_behavior_ros2 llama_behavior_launch.py
```

### 4. 发送图像数据

确保有相机驱动发布 `/camera/image_raw` 话题。测试可用：

```bash
# 用 USB 相机测试
sudo apt install ros-humble-v4l2-camera
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:=[640,480]
```

或用图像发布工具：
```bash
ros2 run image_tools cam2image --ros-args -p frequency:=30.0
```

## 参数配置

### YOLOv11 参数 (`config/yolov11_params.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `onnx_path` | models/yolov11n.onnx | ONNX 模型路径 |
| `trt_path` | models/yolov11n.trt | TensorRT 引擎路径 |
| `input_width` / `input_height` | 640 | 模型输入尺寸 |
| `object_thresh` | 0.5 | 目标置信度阈值 |
| `nms_thresh` | 0.45 | NMS 阈值 |
| `image_topic` | /camera/image_raw | 输入图像话题 |
| `detection_topic` | /yolov11/detections | 检测结果输出话题 |
| `result_image_topic` | /yolov11/result_image | 带框图像输出话题 |
| `publish_result_image` | true | 是否发布带检测框的图像 |
| `class_names` | COCO 80 类 | 类别名称列表 |

### LLaMA 行为识别参数 (`config/llama_params.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `server_url` | http://127.0.0.1:8080/v1/chat/completions | llama.cpp 服务器地址 |
| `prompt_template` | (见下方) | 发送给模型的提示词模板，支持 `{categories_text}` 和 `{valid_ids}` 占位符 |
| `confidence_threshold` | 0.5 | 最低检测置信度 |
| `target_classes` | ["person"] | 只处理的类别（空=全部） |
| `max_detections` | 3 | 每帧最多处理几个检测框 |
| `min_crop_size` | 50 | 最小裁剪尺寸（像素），跳过过小的目标 |
| `detection_topic` | /yolov11/detections | 订阅的检测话题 |
| `image_topic` | /camera/image_raw | 订阅的图像话题 |
| `behavior_topic` | /llama/behavior | 行为描述输出话题 |

## 水域安全行为识别 Prompt

LLaMA 节点内置了一套水域安全行为识别 prompt 模板，启动时自动将行为类别填入占位符，发送给多模态大模型。

### 默认 Prompt 模板

```
你是一个智能水域安全行为识别系统，部署于监控摄像头端，负责实时分析画面中人物的行为并判断安全风险等级。

## 识别原则
仅根据画面中可见的动作姿态进行判断，不做臆测或推断

## 可识别的行为类别
{categories_text}

## 输出要求
请严格按以下 JSON 格式输出，不要包含其他内容：
{
  "behavior_id": "<行为ID>",
  "behavior_label": "<行为英文标签>",
  "description": "<简练行为描述>",
  "severity": "<严重等级: critical/warning/normal>",
  "confidence": <0.7-1.0的置信度>
}

behavior_id 必须是以下之一: {valid_ids}
如果无法确定行为，返回 unknown。
请基于图像内容客观分析，不要臆测。
```

`{categories_text}` 和 `{valid_ids}` 在节点启动时由内置行为类别定义自动填充，无需手动配置。

### 行为类别定义

节点编译时内置了以下 6 种水域安全行为类别（C++ 源码中 `defaultBehaviorClasses()`）：

| ID | 中文标签 | 英文标签 | 严重等级 | 描述 |
|----|---------|---------|---------|------|
| 0 | 溺水 | drowning | critical | 四肢无规律挣扎，有溺水风险。 |
| 1 | 游泳 | swimming | normal | 人员在水中正常游泳。 |
| 2 | 攀爬栏杆 | climbing | warning | 人员攀爬或翻越栏杆。 |
| 3 | 正常行走 | normal_walking | normal | 岸上人员正常行走或站立。 |
| 4 | 正在救援 | waterhelping | normal | 水中人员抱住红色救生圈。 |
| 5 | 在船上 | aboard | normal | 人员在船上或在开船。 |

### 占位符填充后的 Prompt 示例

`{categories_text}` 填充后形如：

```
0: 溺水 (drowning, critical) - 四肢无规律挣扎，有溺水风险。
1: 游泳 (swimming, normal) - 人员在水中正常游泳。
2: 攀爬栏杆 (climbing, warning) - 人员攀爬或翻越栏杆。
3: 正常行走 (normal_walking, normal) - 岸上人员正常行走或站立。
4: 正在救援 (waterhelping, normal) - 水中人员抱住红色救生圈。
5: 在船上 (aboard, normal) - 人员在船上或在开船。
```

`{valid_ids}` 填充后形如：

```
0, 1, 2, 3, 4, 5
```

### 模型输出示例

```json
{
  "behavior_id": "0",
  "behavior_label": "drowning",
  "description": "人员在水中四肢无规律挣扎，头部时沉时浮",
  "severity": "critical",
  "confidence": 0.92
}
```

### 自定义 Prompt

如需修改 prompt，通过 ROS2 参数 `prompt_template` 覆盖即可。保留 `{categories_text}` 和 `{valid_ids}` 占位符以确保行为类别动态注入：

```bash
# 启动时覆盖 prompt_template
ros2 run llama_behavior_ros2 llama_behavior_node \
  --ros-args -p prompt_template:="你的自定义prompt，保留{categories_text}和{valid_ids}"
```

## ROS2 话题一览

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/camera/image_raw` | sensor_msgs/Image | 输入 | 相机原始图像 |
| `/yolov11/detections` | vision_msgs/Detection2DArray | YOLO→LLaMA | 检测结果 |
| `/yolov11/result_image` | sensor_msgs/Image | 输出 | 带检测框的可视化图像 |
| `/llama/behavior` | std_msgs/String | 输出 | LLaMA 生成的行为描述 |

## 调试

```bash
# 查看话题列表
ros2 topic list

# 查看检测结果
ros2 topic echo /yolov11/detections

# 查看行为描述输出
ros2 topic echo /llama/behavior

# 查看带检测框的图像（需安装 rqt_image_view）
ros2 run rqt_image_view rqt_image_view /yolov11/result_image

# 查看节点状态
ros2 node list
ros2 node info /yolov11_tensorrt_node
ros2 node info /llama_behavior_node

# 查看参数
ros2 param list /yolov11_tensorrt_node
ros2 param get /yolov11_tensorrt_node object_thresh
```

## 常见问题

**Q: 编译报错找不到 TensorRT**
```bash
# 指定 TensorRT 安装路径
colcon build --cmake-args -DTENSORRT_ROOT=/usr/lib/x86_64-linux-gnu
# 或
colcon build --cmake-args -DTENSORRT_ROOT=/usr/local/TensorRT-8.6
```

**Q: 编译报错找不到 CUDA**
```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
colcon build
```

**Q: 运行时检测不到目标**
- 确认 `object_thresh` 不要设太高（0.5 是合理值）
- 确认模型输入尺寸是 640x640
- 查看 `/yolov11/result_image` 确认图像是否正常输入

**Q: LLaMA 节点无输出**
- 确认 llama.cpp 服务器已启动且端口 8080 可达
- `curl http://127.0.0.1:8080/v1/chat/completions` 测试连通性
- 确认 `target_classes` 和 YOLO 输出的类别名匹配

**Q: CUDA 架构不匹配导致推理结果异常**
- 根据你的 GPU 修改 `CMakeLists.txt` 中的 `CUDA_GEN_CODE`（见上方表格）
- 修改后重新编译：`colcon build --packages-select yolov11_tensorRT_ros2`

## License

Apache-2.0

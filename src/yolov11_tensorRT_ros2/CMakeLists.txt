cmake_minimum_required(VERSION 3.8)
project(yolov11_tensorRT_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)

# ======================== find dependencies ========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# TensorRT
set(TENSORRT_ROOT /usr/local/TensorRT CACHE PATH "TensorRT root directory")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/)
message(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER})
message(STATUS "Find TensorRT libs: ${TENSORRT_LIBRARY}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

# ======================== CUDA settings ========================
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CUDA_GEN_CODE "-gencode=arch=compute_89,code=sm_89")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --default-stream-per-thread --threads 0 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
enable_language(CUDA)
SET(CMAKE_PREFIX_PATH "/usr/local/cuda")

# ======================== include directories ========================
include_directories(
  include/${PROJECT_NAME}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  src/common
)

# ======================== CUDA kernel library ========================
cuda_add_library(kernels_lib
  src/kernels/image_preprocess.cu
  src/kernels/get_nms_before_boxes.cu
)

# ======================== inference library ========================
add_library(inference_lib
  src/CNN.cpp
  src/postprocess_cuda.cpp
)
target_link_libraries(inference_lib
  kernels_lib
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY}
  ${OpenCV_LIBRARIES}
)

# ======================== ROS2 node executable ========================
add_executable(yolov11_tensorrt_node
  src/yolov11_tensorrt_node.cpp
)
ament_target_dependencies(yolov11_tensorrt_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
  image_transport
)
target_link_libraries(yolov11_tensorrt_node
  inference_lib
  kernels_lib
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY}
  ${OpenCV_LIBRARIES}
)

# ======================== install ========================
install(TARGETS
  yolov11_tensorrt_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY
  models/
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()

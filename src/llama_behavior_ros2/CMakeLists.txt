cmake_minimum_required(VERSION 3.8)
project(llama_behavior_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 14)

# ======================== find dependencies ========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

# CURL
find_package(CURL REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# ======================== include directories ========================
include_directories(
  include/${PROJECT_NAME}
  ${OpenCV_INCLUDE_DIRS}
  ${CURL_INCLUDE_DIRS}
)

# ======================== node executable ========================
add_executable(llama_behavior_node
  src/main.cpp
  src/llama_behavior_node.cpp
)

ament_target_dependencies(llama_behavior_node
  rclcpp
  sensor_msgs
  vision_msgs
  std_msgs
  cv_bridge
  message_filters
)

target_link_libraries(llama_behavior_node
  ${OpenCV_LIBRARIES}
  ${CURL_LIBRARIES}
)

# ======================== install ========================
install(TARGETS
  llama_behavior_node
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

ament_package()

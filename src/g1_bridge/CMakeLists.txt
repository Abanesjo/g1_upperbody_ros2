cmake_minimum_required(VERSION 3.5)
project(g1_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_api REQUIRED)
find_package(unitree_hg REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_executable(g1_bridge_node
  src/g1_bridge_node.cpp
  src/common/motor_crc_hg.cpp
)
add_executable(g1_orchestrator_node
  src/g1_orchestrator_node.cpp
)
add_executable(joy_cmd_vel_node
  src/joy_cmd_vel_node.cpp
)
target_include_directories(g1_bridge_node PRIVATE
  include
  include/common
  ${CMAKE_CURRENT_SOURCE_DIR}/../unitree_ros2_examples/include
)
ament_target_dependencies(g1_bridge_node
  rclcpp
  sensor_msgs
  std_msgs
  unitree_api
  unitree_hg
)
ament_target_dependencies(g1_orchestrator_node
  rclcpp
  sensor_msgs
  std_msgs
)
ament_target_dependencies(joy_cmd_vel_node
  geometry_msgs
  rclcpp
  sensor_msgs
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  g1_bridge_node
  g1_orchestrator_node
  joy_cmd_vel_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

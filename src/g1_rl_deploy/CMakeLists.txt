cmake_minimum_required(VERSION 3.14)
project(g1_rl_deploy)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(g1_rl_deploy_node src/g1_rl_deploy_node.cpp)
ament_target_dependencies(g1_rl_deploy_node rclcpp sensor_msgs geometry_msgs)
target_link_libraries(g1_rl_deploy_node onnxruntime)

install(TARGETS g1_rl_deploy_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/joint_command_gui.py RENAME joint_command_gui DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/motion_playback.py RENAME motion_playback DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/joint_merger.py RENAME joint_merger DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config models motions DESTINATION share/${PROJECT_NAME})

ament_package()

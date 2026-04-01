cmake_minimum_required(VERSION 3.14)
project(g1_rl_deploy)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

set(ONNXRUNTIME_DIR "/opt/onnxruntime")
set(UNITREE_SDK2_DIR "/opt/unitree_robotics")
set(CYCLONEDDS_DIR "/opt/cyclonedds")

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${ONNXRUNTIME_DIR}/include
  ${UNITREE_SDK2_DIR}/include
  ${UNITREE_SDK2_DIR}/include/ddscxx
)

link_directories(
  ${UNITREE_SDK2_DIR}/lib
  ${ONNXRUNTIME_DIR}/lib
)

add_executable(g1_rl_deploy main.cpp)
target_link_libraries(g1_rl_deploy
  unitree_sdk2
  ddsc
  ddscxx
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1.22.0
  pthread
  rt
)

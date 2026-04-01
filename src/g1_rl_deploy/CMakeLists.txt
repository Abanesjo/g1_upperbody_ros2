cmake_minimum_required(VERSION 3.14)
project(g1_rl_deploy)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  /usr/local/include/ddscxx
)

add_executable(g1_rl_deploy main.cpp)
target_link_libraries(g1_rl_deploy
  unitree_sdk2
  ddsc
  ddscxx
  onnxruntime
  pthread
  rt
)

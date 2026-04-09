cmake_minimum_required(VERSION 3.5)
project(g1_human)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/g1_human_node.py
  RENAME g1_human_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

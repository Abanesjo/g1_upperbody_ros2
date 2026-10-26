cmake_minimum_required(VERSION 3.5)
project(g1_human)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY launch config motions
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/g1_human_node.py
  RENAME g1_human_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/human_control_gui.py
  RENAME human_control_gui
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/skeleton_viz_node.py
  RENAME skeleton_viz_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_skeleton_markers
    test/test_skeleton_markers.py
  )
endif()

ament_package()

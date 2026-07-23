cmake_minimum_required(VERSION 3.5)
project(g1_cbf)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/g1_cbf_node.py
  RENAME g1_cbf_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/cmd_vel_cbf_node.py
  RENAME cmd_vel_cbf_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/origin_cmd_vel_publisher_node.py
  RENAME origin_cmd_vel_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/g1_cbf_viz_node.py
  RENAME g1_cbf_viz_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/pelvis_circle_publisher_node.py
  RENAME pelvis_circle_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/pelvis_tf_publisher_node.py
  RENAME pelvis_tf_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/workspace_manager_node.py
  RENAME workspace_manager_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/ghost_publisher_node.py
  RENAME ghost_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_human_capsules
    test/test_human_capsules.py
  )
  ament_add_pytest_test(test_cmd_vel_cbf_enabled
    test/test_cmd_vel_cbf_enabled.py
  )
  ament_add_pytest_test(test_upper_cbf_runtime_gate
    test/test_upper_cbf_runtime_gate.py
  )
  ament_add_pytest_test(test_workspace_manager
    test/test_workspace_manager.py
  )
  ament_add_pytest_test(test_pelvis_circle_workspace
    test/test_pelvis_circle_workspace.py
  )
endif()

ament_package()

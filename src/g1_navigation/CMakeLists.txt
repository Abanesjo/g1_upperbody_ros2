cmake_minimum_required(VERSION 3.8)
project(g1_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ompl REQUIRED)


install(PROGRAMS
  scripts/base_footprint_tf_publisher_node.py
  RENAME base_footprint_tf_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/apriltag_localization_node.py
  RENAME apriltag_localization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_apriltag_localization
    test/test_apriltag_localization.py
  )
  ament_add_pytest_test(test_navigation_launch_contract
    test/test_navigation_launch_contract.py
  )
  ament_add_pytest_test(test_tag_mount_urdf
    test/test_tag_mount_urdf.py
  )
endif()

ament_package()

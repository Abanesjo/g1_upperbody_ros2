cmake_minimum_required(VERSION 3.5)
project(g1_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(g1_cbf_msg REQUIRED)
find_package(pinocchio REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# Installed into /usr/local by the Dockerfile, together with the CycloneDDS
# 0.10.2 libraries it is built against.
find_package(unitree_sdk2 REQUIRED)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

add_library(gravity_compensation STATIC
  src/gravity_compensation.cpp
)
target_include_directories(gravity_compensation PUBLIC include)
target_link_libraries(gravity_compensation PUBLIC pinocchio::pinocchio)

add_library(low_cmd_builder STATIC
  src/low_cmd_builder.cpp
)
target_include_directories(low_cmd_builder PUBLIC include)

# The only code that includes unitree_sdk2. Its CycloneDDS headers differ from
# the ones under the ROS prefix, so they must not reach a target that also
# compiles against ROS. idl/Clock_.* is rosgraph_msgs/Clock for that SDK,
# copied from unitree_mujoco, which publishes it on rt/clock.
add_library(unitree_link STATIC
  src/unitree_link.cpp
  idl/Clock_.cpp
)
target_include_directories(unitree_link
  PUBLIC include
  PRIVATE idl
)
target_link_libraries(unitree_link PRIVATE unitree_sdk2)

# A ROS CycloneDDS (ros-jazzy-rmw-cyclonedds-cpp), where installed, puts its own
# newer libddsc.so.0 on LD_LIBRARY_PATH. An RPATH, unlike the default RUNPATH,
# is searched before LD_LIBRARY_PATH, so it keeps the SDK on the libraries it
# was built against.
get_target_property(UNITREE_DDSC_LOCATION ddsc IMPORTED_LOCATION)
get_filename_component(UNITREE_DDS_LIBRARY_DIR "${UNITREE_DDSC_LOCATION}" DIRECTORY)
function(g1_bridge_use_unitree_dds_libraries target)
  set_target_properties(${target} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${UNITREE_DDS_LIBRARY_DIR}"
  )
  target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endfunction()

add_executable(g1_bridge_node
  src/g1_bridge_node.cpp
)
add_executable(g1_orchestrator_node
  src/g1_orchestrator_node.cpp
)
add_executable(joy_cmd_vel_node
  src/joy_cmd_vel_node.cpp
)
# Republishes unitree_mujoco's rt/clock and rt/pose/pelvis; started by
# g1_navigation/launch/odom_simulation.launch.xml.
add_executable(sim_relay_node
  src/sim_relay_node.cpp
)
ament_target_dependencies(g1_bridge_node
  rclcpp
  sensor_msgs
  std_msgs
)
target_link_libraries(g1_bridge_node
  gravity_compensation
  low_cmd_builder
  unitree_link
)
g1_bridge_use_unitree_dds_libraries(g1_bridge_node)
ament_target_dependencies(sim_relay_node
  geometry_msgs
  rclcpp
  rosgraph_msgs
)
target_link_libraries(sim_relay_node unitree_link)
g1_bridge_use_unitree_dds_libraries(sim_relay_node)
ament_target_dependencies(g1_orchestrator_node
  g1_cbf_msg
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
  sim_relay_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_gravity_compensation
    test/test_gravity_compensation.cpp
  )
  target_compile_definitions(test_gravity_compensation PRIVATE
    G1_BRIDGE_TEST_URDF="${CMAKE_CURRENT_SOURCE_DIR}/test/gravity_pendulum.urdf"
  )
  target_link_libraries(test_gravity_compensation gravity_compensation)

  ament_add_gtest(test_low_cmd_builder
    test/test_low_cmd_builder.cpp
  )
  target_compile_definitions(test_low_cmd_builder PRIVATE
    G1_BRIDGE_TEST_G1_URDF="${CMAKE_CURRENT_SOURCE_DIR}/../g1_description/urdf/g1_29_inspire.urdf"
  )
  target_link_libraries(test_low_cmd_builder
    gravity_compensation
    low_cmd_builder
  )

  # Talks to an in-process fake robot over the loopback interface, on a DDS
  # domain of its own.
  ament_add_gtest(test_unitree_link
    test/test_unitree_link.cpp
    TIMEOUT 60
  )
  target_include_directories(test_unitree_link PRIVATE idl)
  target_link_libraries(test_unitree_link
    low_cmd_builder
    unitree_link
    unitree_sdk2
  )
  g1_bridge_use_unitree_dds_libraries(test_unitree_link)

  ament_add_pytest_test(test_orchestrator_cbf_toggle
    test/test_orchestrator_cbf_toggle.py
    TIMEOUT 20
  )

  ament_add_pytest_test(test_hand_urdfs
    test/test_hand_urdfs.py
    TIMEOUT 20
  )
endif()

ament_package()

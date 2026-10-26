cmake_minimum_required(VERSION 3.5)
project(zed_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keypoint2Df.msg"
  "msg/Keypoint2Di.msg"
  "msg/Keypoint3D.msg"
  "msg/BoundingBox2Df.msg"
  "msg/BoundingBox2Di.msg"
  "msg/BoundingBox3D.msg"
  "msg/Skeleton2D.msg"
  "msg/Skeleton3D.msg"
  "msg/Object.msg"
  "msg/ObjectsStamped.msg"
  DEPENDENCIES std_msgs
)

ament_package()

#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

cd /workspace/ros2_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /workspace/ros2_ws/install/setup.bash


if ! grep -qxF "#Entrypoint Setup" ~/.bashrc; then
    cat <<'EOF' >> ~/.bashrc

#Entrypoint Setup
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=34
export RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
export CYCLONEDDS_URI=file:///workspace/ros2_ws/src/cyclonedds.xml
alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME="cpu"
EOF
fi

exec bash

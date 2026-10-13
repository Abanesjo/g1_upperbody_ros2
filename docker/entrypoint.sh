#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash

cd /workspace/ros2_ws

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source /workspace/ros2_ws/install/setup.bash


if ! grep -qxF "#Entrypoint Setup" ~/.bashrc; then
    cat <<'EOF' >> ~/.bashrc

#Entrypoint Setup
source /opt/ros/jazzy/setup.bash
source /workspace/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION="rmw_zenoh_cpp"
export ZENOH_ROUTER_CONFIG_URI=/workspace/ros2_ws/src/RMW_ZENOH_ROUTER_CONFIG.json5
alias mujoco='export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu'
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export JAX_ENABLE_X64=1
export JAX_PLATFORM_NAME="cpu"
EOF
fi

exec bash

#!/bin/bash
set -e

echo "=== G1 RL Deploy Entrypoint ==="

cd /workspace/src/unitree_ros2/g1_rl_deploy
mkdir -p build && cd build
cmake .. && make -j$(nproc)

echo ""
echo "Build complete. To run:"
echo "  cd /workspace/src/unitree_ros2/g1_rl_deploy/build"
echo "  ./g1_rl_deploy lo"
echo ""

exec bash

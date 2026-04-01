docker build -t g1_upperbody_ros2:v1 .
xhost +local:docker
docker run -it --rm --name g1_upperbody_ros2 --network host \
 --gpus all \
 --runtime=nvidia \
 -e DISPLAY=$DISPLAY \
 -e NVIDIA_DRIVER_CAPABILITIES=all \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v "$(dirname "$(readlink -f "$0")"):/workspace/src/unitree_ros2" \
 g1_upperbody_ros2:v1

FROM osrf/ros:humble-desktop-full

RUN apt update && apt upgrade -y

# ROS2 and build deps
RUN apt install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rosbag2-cpp \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-pinocchio \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-pip \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev \
    tmux

RUN pip3 install numpy==1.26.4 scipy==1.13.1 opencv-contrib-python==4.7.0.72
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
RUN pip3 install osqp "jax[cuda12]" qpax cbfpy
RUN pip3 install onnxruntime

COPY dependencies /workspace/dependencies

#CycloneDDS
RUN cd /workspace/dependencies/cyclonedds && mkdir build && cd build && \
    cmake .. && make -j$(($(nproc) / 2)) && make install && ldconfig

#unitree_sdk2
RUN cd /workspace/dependencies/unitree_sdk2 && mkdir build && cd build && \
    cmake .. && make -j$(($(nproc) / 2)) && make install && ldconfig

#unitree_sdk2_python
# RUN CYCLONEDDS_HOME=/usr/local pip3 install --no-deps /workspace/dependencies/unitree_sdk2_python
RUN CYCLONEDDS_HOME=/opt/ros/humble pip3 install --no-deps /workspace/dependencies/unitree_sdk2_python

#Onnx Runtime C++
COPY dependencies/onnxruntime/lib/ /usr/local/lib/
COPY dependencies/onnxruntime/include/ /usr/local/include/
RUN ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/libonnxruntime.so && ldconfig

RUN mkdir -p /workspace/ros2_ws/src

WORKDIR /workspace/ros2_ws

# ENV CMAKE_PREFIX_PATH=/opt/unitree_robotics:/opt/cyclonedds
# ENV LD_LIBRARY_PATH=/opt/unitree_robotics/lib:/opt/cyclonedds/lib:/opt/onnxruntime/lib

# Workspace setup
WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

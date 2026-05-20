FROM ros:humble

ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt update && apt upgrade -y

RUN apt install ros-humble-desktop -y

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
    sudo \
    tmux

RUN pip3 install numpy==1.26.4 scipy==1.13.1 opencv-contrib-python==4.7.0.72
RUN pip3 install osqp "jax[cuda12]" qpax cbfpy

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
RUN pip3 install --no-deps /workspace/dependencies/dpax

#Onnx Runtime C++
RUN arch="${TARGETARCH:-$(dpkg --print-architecture)}" && \
    if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then \
        ort_src=/workspace/dependencies/onnxruntime_aarch64; \
    else \
        ort_src=/workspace/dependencies/onnxruntime; \
    fi && \
    ort_version="$(cat "$ort_src/VERSION_NUMBER")" && \
    cp -a "$ort_src/include/." /usr/local/include/ && \
    cp -a "$ort_src/lib/." /usr/local/lib/ && \
    ln -sf "libonnxruntime.so.$ort_version" /usr/local/lib/libonnxruntime.so.1 && \
    ln -sf libonnxruntime.so.1 /usr/local/lib/libonnxruntime.so && \
    ldconfig

RUN mkdir -p /workspace/ros2_ws/src

# ENV CMAKE_PREFIX_PATH=/opt/unitree_robotics:/opt/cyclonedds
# ENV LD_LIBRARY_PATH=/opt/unitree_robotics/lib:/opt/cyclonedds/lib:/opt/onnxruntime/lib

# Workspace setup
WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

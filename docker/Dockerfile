FROM ros:jazzy

ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt update && apt upgrade -y

RUN apt install ros-jazzy-desktop -y

# ROS2 and build deps
RUN apt install -y \
    ros-jazzy-rmw-zenoh-cpp \
    ros-jazzy-rosbag2-cpp \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-pinocchio \
    ros-jazzy-joy \
    ros-jazzy-foxglove-bridge \
    ros-jazzy-rqt-tf-tree \
    ros-jazzy-librealsense2* \
    ros-jazzy-rtabmap* \
    ros-jazzy-xacro \
    ros-jazzy-ompl \
    ros-jazzy-launch-pytest \
    ros-jazzy-diagnostic-updater

RUN apt install -y \
    libapr1-dev \
    libaprutil1-dev \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-pip \
    python3-tqdm \
    python3-requests \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev \
    sudo \
    tmux

# numpy and scipy stay pinned for every pip install below. Unpinned, pip picks
# a JAX that needs NumPy 2, which breaks the apt-installed ROS Python packages.
RUN printf 'numpy==1.26.4\nscipy==1.13.1\n' > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# apt installs scipy 1.11.4, which pip cannot uninstall. The pinned scipy goes
# into /usr/local, which Python searches first.
RUN pip3 install --ignore-installed --no-deps scipy==1.13.1
RUN pip3 install numpy==1.26.4 opencv-contrib-python==4.7.0.72
RUN arch="${TARGETARCH:-$(dpkg --print-architecture)}" && \
    if [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then \
        jax_spec="jax==0.6.2"; \
    else \
        jax_spec="jax[cuda12]==0.6.2"; \
    fi && \
    pip3 install osqp "$jax_spec" qpax cbfpy

COPY dependencies /workspace/dependencies

#unitree_sdk2 (installs its own CycloneDDS 0.10.2 libraries and headers into /usr/local)
RUN cd /workspace/dependencies/unitree_sdk2 && mkdir build && cd build && \
    cmake .. && make -j"$(nproc)" && make install && ldconfig

#unitree_sdk2_python
RUN pip3 install --no-deps /workspace/dependencies/unitree_sdk2_python
RUN pip3 install --no-deps /workspace/dependencies/dpax

#inspire hand
WORKDIR /workspace/dependencies/dfx_inspire_service

RUN mkdir build && cd build && cmake .. && cmake --build .

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

# ENV CMAKE_PREFIX_PATH=/opt/unitree_robotics:/opt/cyclonedds
# ENV LD_LIBRARY_PATH=/opt/unitree_robotics/lib:/opt/cyclonedds/lib:/opt/onnxruntime/lib

WORKDIR /workspace/dependencies/Livox-SDK2
# v1.3.1 uses fixed-width integer types without including <cstdint>, which GCC 13 rejects
RUN mkdir build && cd build && cmake .. -DCMAKE_CXX_FLAGS="-include cstdint" && \
    cmake --build . && make install

# Workspace setup
WORKDIR /workspace/ros2_ws

SHELL ["/bin/bash", "-c"]

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]

# g1_upperbody_ros2 
Low level control for a Unitree G1 robot implementing safe human imitation with CBF, along with lower body stability and velocity tracking via an RL policy. 

## Prerequisites

The package works on both x86 and ARM systems but require an NVIDIA GPU. Docker is used to streamline the installation process. If not already, please install docker and NVIDIA's docker container toolkit on your system. Thus, the following are required; 

1. [Docker](https://docs.docker.com/engine/install/)
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Simulator
This is designed to work with the unitree mujoco simulator. Please use the fork from the following repository and follow the setup instructions.

[https://github.com/Abanesjo/unitree_mujoco](https://github.com/Abanesjo/unitree_mujoco)

## Usage

Install and build the docker container using the following command.

```
git clone --recursive https://github.com/Abanesjo/g1_upperbody_ros2
cd docker
docker compose up --build
```

Also grant access to your display for docker applications

```
xhost +local:root
```

Next, enter the docker container

```
docker exec -it g1_upperbody_ros2 bash
```

This will give you access to the workspace within the container. The container already has tmux installed. So you can run

```
tmux
```

and execute the following commands to test the functionality. 

### Motion from data

In one terminal within the container: 
```
ros2 launch g1_cbf bringup.launch.xml rviz:=true simulator:=true
```

And in another terminal: 
```
ros2 launch g1_human g1_human_manual.launch.xml
```

You will see a window below

![image](docs/rviz.png)



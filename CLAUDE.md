# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ROS 2 Jazzy workspace for low-level control of a Unitree G1 (29 DOF): an RL policy for locomotion, control-barrier-function (CBF) safety filters on upper-body joint targets and base velocity, and a joystick-driven orchestrator between the policy and the motors. It drives either the real robot or a MuJoCo simulator. The ROS graph runs on Zenoh (`rmw_zenoh_cpp`); only `g1_bridge_node` talks to the robot or the simulator, and it does so with `unitree_sdk2`, not with ROS topics. In simulation, `sim_relay_node` also listens to the simulator, for `/clock` and `/pose/pelvis`.

## Work only inside Docker

Build, test and launch only inside the `g1_upperbody_ros2` container, never on the host. The host is for editing files and `git`.

```bash
# on the host
cd docker && docker compose up --build -d        # README runs it in the foreground; -d keeps it from blocking
docker logs -f g1_upperbody_ros2                 # startup build is done at "Summary: N packages finished"
xhost +local:root                                # needed for RViz and the PyQt GUIs
docker exec -it g1_upperbody_ros2 bash           # interactive shell in /workspace/ros2_ws
docker exec g1_upperbody_ros2 bash -ic '<cmd>'   # one-off command; the -i matters, see below
cd docker && docker compose down
```

- Only `src/` is bind-mounted (to `/workspace/ros2_ws/src`), so edits under `src/` are live in the container. `dependencies/` is copied and compiled into the image, so changes there or in `docker/` need `docker compose up --build`.
- `docker/entrypoint.sh` runs `rosdep install` and `colcon build --symlink-install` on every container start. `build/`, `install/` and `log/` exist only inside the container and are discarded by `docker compose down`.
- The entrypoint writes the runtime environment into `~/.bashrc`: the ROS and workspace overlays, `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, `ZENOH_ROUTER_CONFIG_URI=/workspace/ros2_ws/src/RMW_ZENOH_ROUTER_CONFIG.json5`, `JAX_ENABLE_X64=1`, `JAX_PLATFORM_NAME=cpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Only interactive shells read that file, so a plain `bash -c` gets none of it. Use `bash -ic`; its "no job control" warning is harmless.
- Nodes find each other only through a Zenoh router, and nothing starts one. Run `ros2 run rmw_zenoh_cpp rmw_zenohd` in the container before launching nodes, using `ros2 topic` or `ros2 node`, or running tests that spawn nodes (`docker exec -d g1_upperbody_ros2 bash -ic 'ros2 run rmw_zenoh_cpp rmw_zenohd'` starts a detached one). It reads `src/RMW_ZENOH_ROUTER_CONFIG.json5` and listens on TCP port 7447.
- To bound a `ros2 topic hz` or `ros2 topic echo` from a script, use `timeout -s INT`; with the default SIGTERM, `ros2 topic hz` kept running.
- The container runs as root, so anything it writes under `src/` is root-owned on the host. Create and edit source files from the host side.
- `cbfpy` and `qpax` are installed from PyPI by the Dockerfile. The checkouts in `dependencies/cbfpy` and `dependencies/qpax` are not what runs.
- The image is Ubuntu 24.04 with Python 3.12 and GCC 13. The Dockerfile pins NumPy 1.26.4 (the apt version the ROS Python packages use), SciPy 1.13.1 and JAX 0.6.2 through `PIP_CONSTRAINT`; newer JAX releases need NumPy 2.

## Build and test

Run these inside the container from `/workspace/ros2_ws`.

```bash
colcon build --symlink-install                             # everything
colcon build --symlink-install --packages-select g1_cbf    # one package
colcon test --packages-select g1_bridge g1_cbf g1_navigation g1_rl_deploy --event-handlers console_direct+
colcon test --packages-select g1_cbf --ctest-args -R test_workspace_manager   # one test target
colcon test-result --verbose
python3 -m pytest -p no:cacheprovider src/g1_cbf/test/test_workspace_manager.py -k test_capture   # one test function
```

- Every package is `ament_cmake`, including the Python ones, so tests run through CTest. Select one with `--ctest-args -R <name>`; `--pytest-args` is ignored.
- A new test file runs only after it is registered in that package's `CMakeLists.txt` (`ament_add_pytest_test` or `ament_add_gtest`). A new Python node likewise needs an `install(PROGRAMS scripts/x_node.py RENAME x_node ...)` entry there.
- Because of `--symlink-install`, edits to existing Python scripts, launch XML and YAML apply on the next launch. C++, `.msg` and CMake changes, and newly added files, need `colcon build`.
- The four packages above hold all of the project's tests. No lint or format step is configured for the `g1_*` packages.
- `-p no:cacheprovider` stops a direct pytest run from leaving a root-owned `.pytest_cache` in `src/`.
- Python tests load node scripts straight from `scripts/` with `importlib`, create nodes with `Class.__new__` (skipping `__init__` and the ROS graph) and stub the rest with `SimpleNamespace`. Decision logic is kept in plain classes so it can be tested this way (`WorkspaceCaptureState`, `MotionPlaybackGate`, `RectangleFollower`). `g1_bridge/test/test_orchestrator_cbf_toggle.py` instead spawns the built orchestrator on a random `ROS_DOMAIN_ID`, so it needs a current build and a running Zenoh router.
- The bridge has two gtests of its own. `test_low_cmd_builder` covers gains per orchestrator state, gravity feed-forward gating and the CRC without ROS or the SDK. `test_unitree_link` runs the SDK link against an in-process fake robot over `lo` on DDS domain 73, away from domain 0, which the robot and the simulator use.
- Some tests pin files, not behaviour: launch-argument defaults (`g1_navigation/test/test_navigation_launch_contract.py`), and URDF structure plus the contents of `g1_bridge/rviz/config.rviz` (`g1_bridge/test/test_hand_urdfs.py`, `g1_navigation/test/test_tag_mount_urdf.py`). Update them together with those files.

## Running

```bash
ros2 launch g1_navigation odom_simulation.launch.xml                      # simulator only, start first: /clock, /pose/pelvis, world -> pelvis TF
ros2 launch g1_cbf bringup.launch.xml simulator:=true rviz:=true          # policy + CBF stack
ros2 launch g1_cbf bringup_manual.launch.xml simulator:=true rviz:=true   # same, plus a slider GUI for upper-body targets
ros2 launch g1_cbf dancing_demo.launch.xml simulator:=true rviz:=true     # same, plus CSV dance playback
ros2 launch g1_human g1_human_manual.launch.xml                           # simulated human colliders with a pose/joint GUI
```

- **These launches command motors.** The bridge writes `rt/lowcmd` with `unitree_sdk2` on the interface named by `network_interface`, which defaults to `lo`, where a simulator on the same machine listens. `simulator` defaults to `false`, which targets the real robot: the bridge first releases Unitree's built-in motion service and exits if that service does not answer, so the two defaults together (`simulator:=false` on `lo`) end in an error and command nothing. `simulator:=true` skips the release and commands whatever listens on that interface, a real robot included. Before launching, confirm with the user whether the target is the simulator or hardware, and which interface.
- Hardware is `simulator:=false network_interface:=<NIC connected to the robot>` (for example `enp130s0`). When `/joint_states` is silent, the bridge receives no `rt/lowstate`: check `network_interface` first, then the bridge's `domain_id` parameter (0, like the robot and the simulator). When a topic has a publisher but other nodes receive nothing, check that the Zenoh router is running.
- Another machine that publishes into this graph (for example the source of `/pose/tag_frame` or `/human/colliders`) needs ROS 2 Jazzy or newer with `rmw_zenoh_cpp` and a Zenoh connection to this router. `rmw_zenoh` on Humble cannot talk to Jazzy.
- The simulator is a separate repo and container (the fork at github.com/Abanesjo/unitree_mujoco, started with its own `./build_and_run.sh`). Its `simulate/config.yaml` has to agree with this repo: `robot` (`g1`, `g1_inspire` or `g1_dex`) with `hand_type`, and `interface`/`domain_id` with the bridge (`lo` and 0). The simulator speaks DDS only, to the bridge; it needs no ROS, no Zenoh and no change for this repo.
- Launch arguments: `network_interface` (see above), `hand_type` (`default`|`inspire`|`dex`, picks `g1_description/urdf/g1_29_<hand_type>.urdf`, which sets the hand mass used for gravity feed-forward; `default` is the stock rubber hand, whose wrist-yaw link plus hand is the 0.254576 kg `G1_STOCK_HAND_MASS_KG` of the MJLab training project), `policy_type` (`mlp`|`gru_mlp`), `wbc`, `gravity`, `area_cbf`, `initial_mode` (`neutral`|`control`), `ghost`, `use_sim_time`.
- A gamepad on `/joy` (calibrated for a Logitech F710) is needed to change orchestrator state; the alternative is `initial_mode:=control`.

## Architecture

### Command pipeline

| # | Node | Topics on the wire | Role |
|---|---|---|---|
| 1 | `joy_cmd_vel_node` (`g1_bridge`, C++) | `/joy` → `/cmd_vel_joy` | Stick axes to a velocity command. |
| 2 | `workspace_cmd_vel_node` (`g1_cbf`) | `/cmd_vel_joy` → `/cmd_vel` | Picks the velocity source: joystick, a fixed rectangle path, or return-to-centre. |
| 3 | `cmd_vel_cbf_node` (`g1_cbf`) | `/cmd_vel` → `/cmd_vel_safe` | Planar CBF-QP that keeps the head inside the workspace circle and away from human capsules. Clipped pass-through until the workspace is enabled. |
| 4 | `g1_cbf_node` (`g1_cbf`) | `/upper_body_targets_unsafe` → `/upper_body_targets` | CBF-QP on 11 upper-body joints: self-collision always, human capsules and workspace circle when enabled. |
| 5 | `g1_rl_deploy_node` (`g1_rl_deploy`, C++) | `/cmd_vel_safe`, `/upper_body_targets`, `/joint_states`, `/imu` → `/joint_commands` | ONNX Runtime policy at 50 Hz, 29 joint position targets. |
| 6 | `g1_orchestrator_node` (`g1_bridge`, C++) | `/joint_commands` → `/joint_commands_safe` | Joystick state machine that gates the policy output. |
| 7 | `g1_bridge_node` (`g1_bridge`, C++) | `/joint_commands_safe` → DDS `rt/lowcmd`; DDS `rt/lowstate` → `/joint_states`, `/imu` | PD gains, CRC and 500 Hz output to the motors through `unitree_sdk2`; feedback to every node above. |

- Topic names in the source are not the names on the wire. `g1_cbf_node` is written against `/joint_commands_unsafe` → `/joint_commands`, `motion_playback` and `joint_command_gui` publish `/joint_commands`, `joy_cmd_vel_node` publishes `/cmd_vel` and `g1_rl_deploy_node` subscribes to `/cmd_vel`. The `<remap>` lines in `g1_cbf/launch/bringup*.launch.xml` and `dancing_demo.launch.xml` produce the table above; there, `/joint_commands` is the policy's 29-joint output. `g1_cbf.launch.xml`, `g1_bridge.launch.xml` and `g1_rl_deploy.launch.xml` run their nodes without these remaps.
- `bringup.launch.xml` starts no publisher for `/upper_body_targets_unsafe`; it is the input for an external motion source. `bringup_manual` adds `joint_command_gui` and `dancing_demo` adds `motion_playback`.
- With `area_cbf:=false`, `bringup.launch.xml` does not start `cmd_vel_cbf_node` and the policy reads `/cmd_vel` directly.
- `wbc` decides who owns the upper body, and it has to match how the policy was trained. With `wbc:=false` (the launch default) `g1_rl_deploy_node` overwrites all 17 upper-body joints of the policy output, as MJLab's `"pd"` upper-body mode (`G1_VEL_UPPER_BODY_CONTROL`) does in training. The 11 CBF-controlled joints come from `/upper_body_targets`, and the wrists, which nothing publishes, hold the default pose. Each target is clamped to within `upper_body_max_delta` (1.5 rad) of the measured position. Upper-body commands are kept inside the range the policy was trained on, 80% of the joint limits (MJLab `UPPER_BODY_JOINT_RANGES`), in two places: `g1_cbf_node` clamps its integrated target (`CONTROLLED_JOINT_TARGET_LIMITS` in `jax_kinematics.py`), so a barrier pushing a joint against its stop cannot wind the target up; `g1_rl_deploy_node` clamps every incoming target (`upper_body_command_limits`) before it becomes an observation or a PD target. A waist-pitch target wound up to −2.04 rad once made the robot fall; in MJLab the same command fells the policy every time, while −0.416 does not. The bridge adds Pinocchio gravity feed-forward on joints 12-28 (if `gravity:=true`). Both `wbc` modes use the same PD gains (`g1_bridge/config/gains.yaml`), the MJLab actuator gains of both upper-body modes. With `wbc:=true` the policy drives all 29 joints and the targets are only an observation; that is MJLab's `"policy"` mode. A `"pd"` policy leaves its wrist outputs untrained (raw values above 10 at a standing pose) and cannot run with `wbc:=true`.
- `rt/lowcmd` and `rt/lowstate` are DDS topics of `unitree_sdk2` (which brings its own CycloneDDS 0.10.2), not ROS topics, so `ros2 topic` does not show them. The G1 uses the SDK's `unitree_hg` types with 35 motor slots; `unitree_go` is for the quadrupeds. The simulator's `rt/clock` and `rt/pose/pelvis` also exist on DDS only; `sim_relay_node` (`g1_bridge`, started by `g1_navigation/odom_simulation.launch.xml`) republishes them as `/clock` and `/pose/pelvis`. It only listens, through `UnitreeLink` without `EnableLowCommand()`, so it has no `rt/lowcmd` writer.
- The bridge is three parts. `low_cmd_builder` turns orchestrator state, targets and gravity torques into a command with its CRC and knows neither ROS nor the SDK. `unitree_link` wraps the SDK behind a pimpl and is the only translation unit that may include SDK headers: their CycloneDDS headers differ from the ones a ROS CycloneDDS installs, and the executable carries an `RPATH` so that the SDK's own DDS libraries are loaded (see `g1_bridge/CMakeLists.txt`). `g1_bridge_node.cpp` is the rclcpp glue. SDK callbacks run on SDK threads, so the state they share with ROS callbacks sits behind `robot_state_mutex_`.

### Orchestrator states and joystick map

`g1_orchestrator_node` is in `neutral` (startup; ramps to a stand pose over 3 s), `control` (forwards the policy output) or `damp` (zero stiffness; latched until the node restarts). The policy node runs in every state; the orchestrator only decides whether its output reaches the bridge. It also publishes the latched gates `/orchestrator/state`, `/cbf/enabled` (true only in `control`), `/cbf/external_enabled` and `/cbf/workspace_enable_request`, and resets the last two to false on every state change. Outside `control`, `g1_cbf_node` skips the QP and echoes the measured joints so re-enabling does not jump, and `workspace_cmd_vel_node` publishes zero velocity.

`sensor_msgs/Joy` indices are read in four files:

| Index | Action | Read in |
|---|---|---|
| button 4 (LB) | enter `damp` (kill switch) | `g1_bridge/src/g1_orchestrator_node.cpp` |
| button 5 (RB) | toggle `neutral` / `control` | same |
| button 2 | toggle the human-collision CBF (`control` only) | same |
| button 3 | toggle the workspace CBF, capturing the current pelvis XY as its centre and the pelvis heading as its x axis (`control` only) | same |
| button 0 | toggle return-to-centre | `g1_cbf/scripts/workspace_cmd_vel_node.py` |
| button 1 | toggle rectangle-path following | same |
| buttons 9 / 10 | stop / start CSV playback | `g1_rl_deploy/scripts/motion_playback.py` |
| axes 1, 0, 3 | forward, lateral, yaw rate | `g1_bridge/src/joy_cmd_vel_node.cpp` |

### CBF implementation (`src/g1_cbf`)

- Library code is in `g1_cbf/g1_cbf/`, nodes in `g1_cbf/scripts/`. `cbf_config.py` holds the two `cbfpy.CBFConfig` subclasses: `G1CollisionCBFConfig` (state is 11 joint positions, control is joint velocity) and `G1CmdVelCBFConfig` (state is the head XY in world, control is body-frame planar velocity).
- `jax_kinematics.py` is hand-written JAX forward kinematics with link offsets copied from the URDF. The CBF never loads the URDF (the `urdf_path` launch parameter is not read by `g1_cbf_node`), so a URDF geometry change has to be mirrored there by hand. The file also owns the collision capsules, the self-collision pairs, `CONTROLLED_JOINTS` and `N_HUMAN_CAPSULES` (11). `g1_human` reuses this FK to build the simulated human.
- Everything is JIT-compiled with fixed shapes. Variable-size inputs are padded into "slots" with a boolean mask, and a masked-out barrier returns `1.0`. Barrier arguments are positional: adding one means changing, in the same order, the `h_1` and `alpha` signatures, the dummy `init_args` in the config, the warm-up call in the node's `__init__` and the per-tick call.
- Both CBF nodes replace `cbf.safety_filter` with their own jitted wrapper around `qpax.solve_qp_elastic_primal` so that `max_iter` can be set.
- Docstrings say the QP runs on the GPU, but the container sets `JAX_PLATFORM_NAME=cpu`.
- `config/params.yaml` is one file with a section per node name. Quantities several nodes use, such as `world_circle_radius` and `head_collider_radius`, are set separately in each section.

### Localization and frames

Human avoidance, the workspace circle and path following need the TF `world → pelvis`, and the bringup launches do not provide it. Start exactly one source from `g1_navigation`:

- `odom_simulation.launch.xml` (simulator): `/pose/pelvis` from the MuJoCo fork, republished by its `sim_relay_node`, becomes `world → base_footprint`, plus a static `base_footprint → pelvis`. The same node publishes `/clock`. The launch takes `network_interface` and `domain_id`, which must match the simulator's `config.yaml`, like the bridge's.
- `apriltag_localization.launch.xml` (hardware): an externally tracked tag pose on `/pose/tag_frame`, composed with the URDF's `tag_mount_frame ← pelvis` at the same timestamp. `tag_mount_frame` exists in the three `g1_29_<hand_type>.urdf` files but not in `g1_29dof.urdf`.
- `rtabmap.launch.xml` (hardware): ICP odometry from the Livox MID-360.

`workspace_manager_node` adds `world → workspace`. When the workspace CBF is requested it captures the pelvis XY as the origin and the pelvis heading (yaw only, roll and pitch dropped) as the frame's x axis, and increments `generation` in the latched `/cbf/workspace_state`. `workspace_cmd_vel_node` works in the `workspace` frame, so the rectangle path starts straight ahead of the robot and return-to-centre turns it back to the captured heading; it restarts its path when the generation changes. The CBF nodes read only the translation, since the workspace is a circle. `/human/colliders` (`g1_cbf_msg/CapsuleArray`, at most 11 capsules) may arrive in any TF frame and is transformed to `world` at its stamp.

### Conventions that are easy to break

- **Freshness gates.** When a filter needs `world → pelvis` and it is missing, stale or future-dated, the upper-body CBF holds the measured joints, the velocity CBF zeroes linear velocity and a workspace capture is rejected. A stamp up to 20 ms ahead of the node clock still counts as fresh (`TfPoseLookup` `future_tolerance_sec`): under simulation time, `/pose/pelvis` runs a few milliseconds ahead of the `/clock` a Python node has processed. Stale `/human/colliders` drops the human constraints for that tick. Keep these behaviours when editing those nodes.
- **One clock per session.** The gates compare message and TF stamps with the node clock (`tf_stale_timeout_sec`, `human_timeout_sec`), and the simulator stamps `/pose/pelvis` in simulation time. In simulation, anything that needs `world → pelvis` therefore requires `use_sim_time:=true` on the bringup and `g1_human` launches too; they default to `false`, `odom_simulation.launch.xml` to `true`. `/clock` exists only while `odom_simulation.launch.xml` runs, so nodes on simulation time stand still without it; start it before the bringup. The orchestrator takes the start of its 3 s stand-up ramp from its first published command, which needs `/clock`. It used to take it from the first `/joint_states`, which could arrive before the first `/clock`, and the ramp then ended at once.
- **QoS.** Streams (`/joint_states`, `/joint_commands*`, `/upper_body_targets*`, `/cmd_vel*`, `/human/colliders`) are BEST_EFFORT, VOLATILE, depth 1. State topics (`/orchestrator/state`, `/cbf/enabled`, `/cbf/external_enabled`, `/cbf/workspace_enable_request`, `/cbf/workspace_state`) are RELIABLE, TRANSIENT_LOCAL, depth 1. Match the existing profile when adding a publisher or subscriber.
- **Joint order.** The 29-joint motor order (legs 0-11, waist 12-14, left arm 15-21, right arm 22-28; index map at the top of `g1_rl_deploy/config/config.yaml`) is hard-coded as `JOINT_NAMES` in `g1_bridge_node.cpp`, `g1_orchestrator_node.cpp`, `g1_rl_deploy_node.cpp` and several Python scripts and tests. The 11 CBF-controlled joints (waist, shoulders and elbows; wrists excluded) are repeated in `jax_kinematics.py`, `motion_playback.py` and the GUIs. `g1_rl_deploy_node.cpp` instead lists all 17 upper-body joints (`UPPER_BODY_INDICES`). Change them together.
- **Policy contract.** `g1_rl_deploy_node` validates the ONNX file at load: input `obs` of 115 floats (angular velocity 3, projected gravity 3, velocity command 3, gait phase 2, joint position 29, joint velocity 29, last action 29, upper-body targets 17) and output `actions` of 29; `gru_mlp` adds `h_in`/`h_out` of shape `[1,1,256]`. `default_joint_pos` and `action_scale` in `g1_rl_deploy/config/config.yaml` must match the trained policy (training configs are in `models/<type>/params/`). The orchestrator's neutral pose (`DEFAULT_JOINT_POS`) is a separate constant with different arm angles.
- **Model paths.** `config.yaml` selects `models_isaaclab/gru_mlp_pd/policy.onnx` for `gru_mlp` and `models_rubber/mlp/policy.onnx` for `mlp`; the node's compiled-in fallback is `models_rubber/<type>/policy.onnx`. `models_isaaclab/` holds the IsaacLab (`g1_vel_isaaclab`) GRU policies: `gru_mlp_pd` is `G1-Vel-GRU-PD` run `g1_vel_gru_pd/2026-10-23_01-53-23_pd_runpod`, checkpoint 6700, for `wbc:=false`; `gru_mlp_policy` is `G1-Vel-GRU` run `g1_vel_gru/2026-10-22_21-34-24_policy_runpod`, checkpoint 8800, for `wbc:=true`. Both walked without falling in `unitree_mujoco` through this stack. `models_rubber/gru_mlp` is the MJLab run `g1_vel_gru/2026-10-14_18-23-08_rubber_hand_pd`, trained in `"pd"` upper-body mode. `models/` holds older exports (`run_name: v2`) trained in `"policy"` mode. `models_rubber/mlp` is byte-identical to `models/mlp`, and `models_old/` holds older alternates. Model and motion-file paths are absolute container paths under `/workspace/ros2_ws/install/`.
- **Vendored code.** Leave `dependencies/` and the submodules `src/livox_ros2` and `src/realsense-ros` alone unless the task is about them. `g1_bridge/idl/Clock_.*` (`rosgraph_msgs/Clock` for the SDK) is a copy of `simulate/idl/Clock_.*` in the simulator fork; both ends have to use the same type.
- `references/` holds symlinks to sibling checkouts (the simulator fork and the MJLab training project). They may dangle on another machine.

## Style, commits and pull requests

- Python uses 4 spaces; node scripts are `scripts/*_node.py`, launch files are `*.launch.xml`, and parameters live in YAML under each package's `config/`. C++ indentation differs by file (2 spaces in `g1_bridge`, 4 in `g1_rl_deploy`); match the file.
- Commit subjects are short, imperative and lowercase (`fix sphere colliders`, `tune params`), optionally prefixed with the package (`g1_cbf: add capsule collision test`).
- Pull requests state the runtime impact, list the commands tested, call out robot-safety or hardware assumptions, and include RViz captures for visualization changes.

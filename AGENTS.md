# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 Humble workspace for Unitree G1 upper-body control. Active packages live under `src/`; vendored code and binary dependencies live under `dependencies/`.

- `src/g1_cbf`: Python CBF safety filter nodes, config, and launch files.
- `src/g1_bridge`: Python bridge node, gains config, and RViz setup.
- `src/g1_rl_deploy`: C++ ONNX deploy node plus helper scripts, model files, motions, and launch config.
- `src/g1_human`: human-motion simulator scripts, config, launch files, and CSV motions.
- `src/g1_description`: URDF and mesh assets.
- `src/interfaces`: ROS message packages for Unitree and CBF messages.
- `src/unitree_ros2_examples`: C++ example clients and control demos.

Avoid editing `dependencies/` unless the change is explicitly for vendored code.

## Build, Test, and Development Commands

- `./build_and_run.sh`: builds the Docker image and opens an interactive ROS workspace container with GPU, host networking, and X11.
- `colcon build --symlink-install`: builds all ROS packages from the workspace root.
- `source install/setup.bash`: overlays the built workspace before launching nodes.
- `colcon test --event-handlers console_direct+`: runs available ROS tests and ament lint checks.
- `ros2 launch g1_cbf bringup.launch.xml`: launches the CBF bringup flow. Other launch files follow the same pattern, for example `ros2 launch g1_bridge g1_bridge.launch.xml`.

The container entrypoint also sets CycloneDDS defaults through `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and `CYCLONEDDS_URI`.

## Coding Style & Naming Conventions

Use 4-space indentation for Python and follow existing ROS naming patterns: scripts end in `_node.py`, launch files end in `.launch.xml`, and config stays in YAML under each package's `config/` directory. C++ code uses lowercase CMake targets and the standard specified per package. Keep ROS package names lowercase with underscores.

Do not commit generated `build/`, `install/`, `log/`, `.cache/`, `__pycache__/`, or `.vscode/` content.

## Testing Guidelines

There are no project-local test directories under `src/` at present. Interface and example packages use `ament_lint_auto` when `BUILD_TESTING` is enabled. For new logic, add package-local ROS tests under `test/` and name files by behavior, for example `test_cbf_config.py` or `test_joint_limits.cpp`. Run `colcon test` after changing messages, CMake, launch files, or runtime nodes.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lowercase commit subjects such as `fix sphere colliders`, `add human simulator`, and `switch to jax`. Keep commits focused and mention the package when helpful, for example `g1_cbf: add capsule collision test`.

Pull requests should describe the runtime impact, list tested commands, call out robot-safety or hardware assumptions, and include screenshots or RViz captures for visualization changes.

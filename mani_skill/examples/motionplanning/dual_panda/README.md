# Dual Panda Motion Planning

This directory contains the custom bimanual motion-planning stack used for the
dual Franka Panda tasks in Colosseum V2. The planner is integrated with
ManiSkill environments and provides scripted demonstrations for coordinated
dual-arm manipulation.

The Colosseum V2 task definitions live in:

```text
mani_skill/envs/tasks/tabletop/colosseum_v2/
```

The Dual Panda robot agent is defined in:

```text
mani_skill/agents/robots/panda/dual_panda.py
```

## Directory Layout

```text
dual_panda/
├── bimanual_planner.py      # Core MPlib/OMPL/Pinocchio planner
├── motionplanner.py         # ManiSkill-facing solver and trajectory executor
├── solutions/               # Per-environment scripted demonstrations
├── media/                   # Demo MP4s
└── README.md
```

## Planner Overview

The planner is split into two layers:

- `bimanual_planner.py` builds the robot model and planning world, handles
  dual-arm IK, self/environment collision checks, OMPL/RRTConnect planning,
  screw-style Cartesian motion, constrained object transport, path smoothing,
  and TOPPRA time parameterization.
- `motionplanner.py` connects the planner to ManiSkill/SAPIEN, converts between
  simulator and planner joint ordering, executes trajectories through
  `pd_joint_pos`, manages per-arm gripper state, and exposes high-level motion
  primitives used by the solution scripts.

Supported high-level primitives include:

- Single-arm pose motion while keeping the other arm fixed
- Dual-arm pose-pair motion with RRTConnect
- Single-arm and dual-arm screw motions
- Per-arm and synchronized gripper open/close control

## Running a Demo


From the repository root, you can run one of the bimanual solutions through the
shared Panda motion-planning runner:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
  --env-id DualArmPickCube-v1 \
  --num-traj 1 \
  --distraction-set none \
  --obs-mode none \
  --vis
```

The runner can also record videos and trajectories:

```bash
python -m mani_skill.examples.motionplanning.panda.run \
  --env-id DualArmLiftTray-v1 \
  --num-traj 1 \
  --distraction-set none \
  --obs-mode rgb \
  --save-video \
  --traj-name trajectory
```

Most solution files can also be run directly for interactive debugging:

```bash
python -m mani_skill.examples.motionplanning.dual_panda.solutions.bimanual_pass_cube
```

## Demo GIFs

Add demo GIFs under a future `media/` folder using the filenames referenced in
the table below. The Markdown image placeholders will render automatically once
the GIFs are added.

## Environment Catalogue

| Environment | Solution script | Task summary | Demo |
| --- | --- | --- | --- |
| `DualArmPickCube-v1` | `solutions/bimanual_pass_cube.py` | Pick and pass a cube between the two arms. | <video src="https://github.com/user-attachments/assets/b9d2fd23-298d-4be4-8317-f2c0ce918478" width="100%" controls></video>|
| `DualArmPickBottle-v1` | `solutions/bimanual_pass_bottle.py` | Pick and pass a bottle between the two arms. | <video src="https://github.com/user-attachments/assets/71d735db-36c5-4460-95dd-ca4a042470c4" width="100%" controls></video>|
| `DualArmPourPot-v1` | `solutions/bimanual_pour_pot.py` | Coordinate both arms to grasp, move, and pour with a pot. | <video src="https://github.com/user-attachments/assets/fb1280ec-f0af-4ea7-96a1-15c4d4a4f7b6" width="100%" controls></video>|
| `DualArmLiftPot-v1` | `solutions/bimanual_lift_pot.py` | Lift and transport a pot using both grippers. | <video src="https://github.com/user-attachments/assets/c60c718b-20e1-4245-b01f-fd2d18265102" width="100%" controls></video>|
| `DualArmLiftTray-v1` | `solutions/bimanual_lift_tray.py` | Lift and transport a tray with synchronized two-arm grasps. | <video src="https://github.com/user-attachments/assets/0451351b-f98a-4828-8ce5-7892d9f45cb7" width="100%" controls></video>|
| `DualArmDrawerOpen-v1` | `solutions/bimanual_drawer_open.py` | Grasp drawer handles and open the drawer with coordinated arm motion. | <video src="https://github.com/user-attachments/assets/5e8edb10-1dc5-4c4f-ae25-e99e5578e0e6" width="100%" controls></video>|
| `DualArmDrawerPlace-v1` | `solutions/bimanual_drawer_place.py` | Open a drawer and place an object inside with two-arm coordination. | <video src="https://github.com/user-attachments/assets/c5053dec-8501-4c9b-8ac7-e2fc938900e4" width="100%" controls></video>|
| `DualArmPenCap-v1` | `solutions/bimanual_pen_cap.py` | Coordinate both arms to manipulate a pen and cap. | <video src="https://github.com/user-attachments/assets/6ae8abc2-b330-4c18-9262-168defd027a2" width="100%" controls></video>|
| `DualArmPushBox-v1` | `solutions/bimanual_push_box.py` | Push a box using synchronized end-effector motion. | <video src="https://github.com/user-attachments/assets/0f896c0c-c632-46e6-9db6-6c490766e219" width="100%" controls></video>|
| `DualArmStackCube-v1` | `solutions/bimanual_stack_cubes.py` | Stack cubes with coordinated bimanual pick-and-place. | <video src="https://github.com/user-attachments/assets/edd618ee-89b0-421c-988c-fc5fc0b02445" width="100%" controls></video>|
| `DualArmStack3Cube-v1` | `solutions/bimanual_stack3cubes.py` | Build a three-cube stack using both arms. | <video src="https://github.com/user-attachments/assets/96dc03af-4bbb-4473-a871-6eb94f64f8d0" width="100%" controls></video>|
| `DualArmThreading-v1` | `solutions/bimanual_threading.py` | Coordinate both arms for a threading-style manipulation task. | <video src="https://github.com/user-attachments/assets/70c5ed19-6004-43e9-907a-6d6bb1aa877c" width="100%" controls></video>|

## Implementation Notes

- The planner uses the fixed-base `dual_panda_table.urdf` robot model.
- The active robot consists of two Panda arms with independent TCP links and
  grippers, exposed by the `dual_panda` ManiSkill agent.
- Solution scripts generally instantiate `DualPandaMotionPlanningSolver`, reset
  the environment with a seed, compute task-specific grasp or target poses, and
  execute a sequence of RRTConnect, screw-motion, and gripper primitives.
- For data generation, use the shared motion-planning runner so trajectories and
  videos are recorded through ManiSkill's `RecordEpisode` wrapper.

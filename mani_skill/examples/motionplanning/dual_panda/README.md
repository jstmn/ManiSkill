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
| `DualArmPickCube-v1` | `solutions/bimanual_pass_cube.py` | Pick and pass a cube between the two arms. | ![DualArmPickCube demo placeholder](media/DualArmPickCube.mp4) |
| `DualArmPickBottle-v1` | `solutions/bimanual_pass_bottle.py` | Pick and pass a bottle between the two arms. | ![DualArmPickBottle demo placeholder](media/DualArmPassBottle.mp4) |
| `DualArmPourPot-v1` | `solutions/bimanual_pour_pot.py` | Coordinate both arms to grasp, move, and pour with a pot. | ![DualArmPourPot demo placeholder](media/DualArmPourPot.mp4) |
| `DualArmLiftPot-v1` | `solutions/bimanual_lift_pot.py` | Lift and transport a pot using both grippers. | ![DualArmLiftPot demo placeholder](media/DualArmLiftPot.mp4) |
| `DualArmLiftTray-v1` | `solutions/bimanual_lift_tray.py` | Lift and transport a tray with synchronized two-arm grasps. | ![DualArmLiftTray demo placeholder](media/DualArmLiftTray.mp4) |
| `DualArmDrawerOpen-v1` | `solutions/bimanual_drawer_open.py` | Grasp drawer handles and open the drawer with coordinated arm motion. | ![DualArmDrawerOpen demo placeholder](media/DualArmDrawerOpen.mp4) |
| `DualArmDrawerPlace-v1` | `solutions/bimanual_drawer_place.py` | Open a drawer and place an object inside with two-arm coordination. | ![DualArmDrawerPlace demo placeholder](media/DualArmDrawerPlace.mp4) |
| `DualArmPenCap-v1` | `solutions/bimanual_pen_cap.py` | Coordinate both arms to manipulate a pen and cap. | ![DualArmPenCap demo placeholder](media/DualArmPenCap.mp4) |
| `DualArmPushBox-v1` | `solutions/bimanual_push_box.py` | Push a box using synchronized end-effector motion. | ![DualArmPushBox demo placeholder](media/DualArmPushBox.mp4) |
| `DualArmStackCube-v1` | `solutions/bimanual_stack_cubes.py` | Stack cubes with coordinated bimanual pick-and-place. | ![DualArmStackCube demo placeholder](media/DualArmStackCube.mp4) |
| `DualArmStack3Cube-v1` | `solutions/bimanual_stack3cubes.py` | Build a three-cube stack using both arms. | ![DualArmStack3Cube demo placeholder](media/DualArmStack3Cubes.mp4) |
| `DualArmThreading-v1` | `solutions/bimanual_threading.py` | Coordinate both arms for a threading-style manipulation task. | ![DualArmThreading demo placeholder](media/DualArmThreading.mp4) |

## Implementation Notes

- The planner uses the fixed-base `dual_panda_table.urdf` robot model.
- The active robot consists of two Panda arms with independent TCP links and
  grippers, exposed by the `dual_panda` ManiSkill agent.
- Solution scripts generally instantiate `DualPandaMotionPlanningSolver`, reset
  the environment with a seed, compute task-specific grasp or target poses, and
  execute a sequence of RRTConnect, screw-motion, and gripper primitives.
- For data generation, use the shared motion-planning runner so trajectories and
  videos are recorded through ManiSkill's `RecordEpisode` wrapper.

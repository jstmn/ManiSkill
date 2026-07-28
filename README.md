# ColosseumV2

This repository contains the code for the ColosseumV2 project.

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2605.27759)

## Installation

```bash
conda create -n colosseum_v2 python=3.10
conda activate colosseum_v2
pip install torch
pip install -e .

# download the assets
python -m mani_skill.utils.download_asset all

# install the dependencies for the ACT model
pip install -e examples/baselines/act_clip

# (optional) install torch for cuda 12
# python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```



## Code structure

**Files and classes relevant to ColosseumV2**

| Path | Role |
|------|------|
| `mani_skill/envs/tasks/tabletop/colosseum_v2/` | Task environments (single-arm + bimanual). Each task inherits from `ColosseumV2Env`. |
| `mani_skill/envs/tasks/tabletop/colosseum_v2/colosseum_v2_core.py` | Defines the base environment class `ColosseumV2Env`. This class contains the core logic which applies visual and physical perturbations (MO/RO color & texture, camera pose, etc). |
| `mani_skill/envs/tasks/tabletop/colosseum_v2/perturbation_set.py` | Defines the `PerturbationSet` class which is used to configure which perturbations are enabled in an environment. Note that a set of preset perturbation sets are defined in the `PERTURBATION_SETS` dictionary in this file. |
| `mani_skill/examples/motionplanning/panda/run.py` | This script is used to generate motion-planning demonstration trajectories. You may optionally pass a `--perturbation-set` here to enable a preset while collecting trajectories. Note that this should *not* be done when evaluating a VLA on the ColosseumV2 paper. Perturbations sets should only be enabled during evaluation |

**Tasks**:

- Single-arm: `PlaceBookInShelf-v1`, `CookItemInPan-v1`, `PickSodaFromCabinet-v1`, `PickDishFromRack-v1`, `StackCubeColosseumV2-v1`, `PlaceDishInRack-v1`, `LiftPegUprightColosseumV2-v1`, `RotateArrow-v1`, `PegInsertionSideColosseumV2-v1`, `PlugChargerColosseumV2-v1`, `HammerNail-v1`, `ScoopBanana-v1`, `OpenDrawer-v1`, `OpenCabinet-v1`, `PlaceCubeInDrawer-v1`, `RaiseCube-v1`
- Bimanual: `DualArmDrawerOpen-v1`, `DualArmPickCube-v1`, `DualArmPickBottle-v1`, `DualArmLiftPot-v1`, `DualArmLiftTray-v1`, `DualArmPushBox-v1`, `DualArmPourPot-v1`, `DualArmThreading-v1`, `DualArmPenCap-v1`, `DualArmDrawerPlace-v1`, `DualArmStackCube-v1`, `DualArmStack3Cube-v1`

**Configure an environment with a perturbation** by passing a `PerturbationSet` (or a dict of `*_cfg` fields) via `perturbation_set` when creating the env. Note that a set of preconfigured perturbation sets are defined in `PERTURBATION_SETS` in `perturbation_set.py` - these are the exact perturbation sets used in the ColosseumV2 paper. Example code:

```python
import gymnasium as gym
import mani_skill.envs
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import PERTURBATION_SETS

env = gym.make(
    "RaiseCube-v1",
    obs_mode="rgb",
    control_mode="pd_joint_pos",
    perturbation_set=PERTURBATION_SETS["TABLE_COLOR"],
    _env_id="RaiseCube-v1",
)
env.reset(seed=0, options={"reconfigure": True})  # reconfigure resamples distractors
```

**Generate demonstration trajectories** via motion planning (`run.py`). Note that `--perturbation-set` is required and must be one of the preset names above (case-insensitive; looked up as `PERTURBATION_SETS[name.upper()]`). Note that 'none' should be provided if you are not using any perturbations.

```bash
python mani_skill/examples/motionplanning/panda/run.py \
    --env-id RaiseCube-v1 \
    --num-traj 5 \
    --perturbation-set table_color \
    --num-procs 1 \
    --obs-mode rgb \
    --reward-mode none \
    --only-count-success \
    --traj-name trajectory
```


**Combine perturbation sets**, with the `PerturbationSet.merge` method:

```python
from mani_skill.envs.tasks.tabletop.colosseum_v2.perturbation_set import (
    ColorRange,
    PerturbationSet,
    PERTURBATION_SETS,
)

# Merge two presets (enabled factors must not overlap)
ds = PerturbationSet.merge([
    PERTURBATION_SETS["MO_COLOR"],
    PERTURBATION_SETS["TABLE_COLOR"],
])

# Or configure directly
ds = PerturbationSet(
    MO_color_cfg={"color_range": ColorRange(low=(0, 0, 0, 1), high=(1, 1, 1, 1))},
    table_color_cfg={"color_range": ColorRange(low=(0, 0, 0, 1), high=(1, 1, 1, 1))},
)
```

**Note:** Some tasks disable certain factors via `DISABLED_PERTURBATION_FACTORS`. If you request a disabled factor (and are not using `all`), env creation raises `PerturbationFactorDisabledError`.

**Create a custom perturbation set** by defining a `PerturbationSet` and passing it to the environment. Example code:

```python

# Move the camera to the right by [5-10] cm in x direction
ps_camera_in_positive_x = PerturbationSet(
    camera_pose_cfg = {
        "rpy_range": ((0, 0, 0), (0, 0, 0)),       # rotation doesn't change
        "xyz_range": ((0.05, 0, 0), (0.1, 0, 0)),
    },
)

# Add many distractors to the environment
ps_many_distractors = PerturbationSet(
    distractor_object_cfg={
        "n_distractors": 10,
        "x_lims": (-0.5, 0.5),
        "y_lims": (-0.5, 0.5),
    },
)
```



## Train a multi task language conditioned ACT model

https://storage.googleapis.com/bucket-colosseum-v2/trajectory__cv2-full__pd_ee_delta_pose__100.h5

``` bash

# Download the multitask datasets.
# - trajectory__cv2-full__pd_ee_delta_pose__100: Single-Arm dataset
# - trajectory__cv2-full__pd_joint_pos__100: Bimanual dataset
curl --create-dirs -o demos/trajectory__cv2-full__pd_ee_delta_pose__100.h5 https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_ee_delta_pose__100.h5
curl --create-dirs -o demos/trajectory__cv2-full__pd_ee_delta_pose__100.json https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_ee_delta_pose__100.json
curl --create-dirs -o demos/trajectory__cv2-full__pd_joint_pos__100.h5 https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_joint_pos__100.h5
curl --create-dirs -o demos/trajectory__cv2-full__pd_joint_pos__100.json https://storage.googleapis.com/colosseum-v2-public/trajectory__cv2-full__pd_joint_pos__100.json

# (OPTIONAL) Alternatively, create the datasets from scratch. Note: if you see 'Directory not found: demos/TASK-ID/demos__cv2-full_pd_joint_pos__100.h5', simply rerun the script.
./scripts/data_generation/motionplanning_colosseum_v2_bimanual.sh
./scripts/data_generation/motionplanning_colosseum_v2_single_arm.sh

# Single-Arm
python examples/baselines/act_clip/train_rgbd.py \
    --seed 1 --perturbation-set none --demo-path demos/trajectory__cv2-full__pd_ee_delta_pose__100.h5 --sim-backend physx_cuda --num_eval_envs 1 --exp-name=SingleArm_ACT_Clip --control_mode pd_ee_delta_pose --track --batch_size 256 --eval-freq 10000 --max-episode-steps 300 --log-freq 1000 --total-iters 300000 --lr 0.0001 --kl_weight 10.0 --num_queries 30 --hidden_dim 512 --dim_feedforward 1600 --enc_layers 4 --dec_layers 7 --save_freq 10000 --num_eval_episodes 5 --is_multi_task True --target_num_cams 3 --internal_instruction

# Bimanual
python examples/baselines/act_clip/train_rgbd.py \
    --seed 1 --perturbation-set none --demo-path demos/trajectory__cv2-full__pd_joint_pos__100.h5 --sim-backend physx_cuda --num_eval_envs 1 --exp-name=Bimanual_ACT_Clip --control_mode pd_joint_pos --track --batch_size 2 --eval-freq 10000 --max-episode-steps 300 --log-freq 1000 --total-iters 300000 --lr 0.0001 --kl_weight 10.0 --num_queries 30 --hidden_dim 512 --dim_feedforward 1600 --enc_layers 4 --dec_layers 7 --save_freq 10000 --num_eval_episodes 5 --is_multi_task True --target_num_cams 3 --internal_instruction

# Evaluate the models
# Run the entire evaluation loop:
bash examples/baselines/act_clip/eval_rgbd_loop.sh

# Alternatively, run a single evaluation:
# Single-Arm
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path /PATH/TO/CHECKPOINT/best_eval_success_once.pt \
    --control-mode "pd_ee_delta_pose" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 3 \
    --num-eval-episodes 200 \
    --num-eval-envs 34 \
    --max-episode-steps-from-lookup \
    --internal-instruction \
    --perturbation-set "BLANK" \
    --results-path $LOGS_DIR/results_single_arm__table.csv

# Bimanual
python examples/baselines/act_clip/eval_rgbd.py \
    --checkpoint-path /PATH/TO/CHECKPOINT/best_eval_success_once.pt \
    --control-mode "pd_joint_pos" \
    --no-include-depth \
    --sim-backend "physx_cuda" \
    --is-multi-task True \
    --target-num-cams 4 \
    --num-eval-episodes 200 \
    --num-eval-envs 34 \
    --max-episode-steps-from-lookup \
    --internal-instruction \
    --perturbation-set "BLANK" \
    --results-path $LOGS_DIR/results_bimanual_act.csv
```



## Finetune and evaluate Pi0.5

Instructions for how to finetune and evaluate Pi0.5 are available at [Geeksongs/lerobot_colosseum_v2](https://github.com/Geeksongs/lerobot_colosseum_v2)


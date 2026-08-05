#!/bin/bash
# Generate k movies per perturbation for RaiseCube, using the hardware-figure
# perturbation subset (see HARDWARE_ROWS RaiseCube keys in figure_hardware.py).
# One env at a time, rolled out k times.

K=10

# Matches RaiseCube keys in scripts/colosseum_v2_paper/figure_hardware.py HARDWARE_ROWS.
PERTURBATIONS=(
    none
    # light_color
    # mo_size
    # background_color
    # table_color
    # mo_color
    # distractor_object
    # language_none
)

CHECKPOINT="checkpoints/hyeonho_mar17/hyeonho_mar17_act_clip_single_arm_3cameras_15687623_checkpoints_best_eval_success_once.pt"

for pert in "${PERTURBATIONS[@]}"; do
    echo "=== RaiseCube-v1 / ${pert} (1 env x ${K} episodes) ==="
    python examples/baselines/act_clip/eval_rgbd.py \
        --checkpoint-path "$CHECKPOINT" \
        --control-mode "pd_ee_delta_pose" \
        --no-include-depth \
        --sim-backend "physx_cpu" \
        --is-multi-task True \
        --target-num-cams 3 \
        --num-eval-episodes "$K" \
        --num-eval-envs 1 \
        --max-episode-steps-from-lookup \
        --internal-instruction \
        --capture-video \
        --no-metrics-on-video \
        --env-id "RaiseCube-v1" \
        --perturbation-set "$pert"
done

echo "Done. Videos under: ${CHECKPOINT%.pt}__videos/"

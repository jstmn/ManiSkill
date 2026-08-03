#!/usr/bin/env bash
# Collect 5 motion-planning demos per (task, perturbation), extract combined
# videos, and move them into videos/<env_id>/<perturbation>/.
# Already-completed (env, perturbation) pairs with 5 videos are skipped.
set -euo pipefail

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate colosseum_v2

ENV_IDS=(
  "RaiseCube-v1"
  "RotateArrow-v1"
  "LiftPegUprightColosseumV2-v1"
  "PickDishFromRack-v1"
  "PickSodaFromCabinet-v1"
)

# Friendly labels -> PERTURBATION_SETS keys (case-insensitive; looked up as .upper())
PERTURBATION_SETS=(
  "none"
  "light_color"
  "MO_size"
  "background_color"
  "table_color"
  "MO_color"
  "distractor_object"
)

NUM_TRAJ=5
VIDEOS_DIR="videos"

# Env-level DISABLED_PERTURBATION_FACTORS that conflict with requested presets.
is_unsupported_combo() {
  local env_id="$1"
  local perturbation="$2"
  case "${env_id}:${perturbation}" in
    PickDishFromRack-v1:MO_size|PickSodaFromCabinet-v1:MO_size)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

mkdir -p "${VIDEOS_DIR}"

for ENV_ID in "${ENV_IDS[@]}"; do
  for PERTURBATION_SET in "${PERTURBATION_SETS[@]}"; do
    TRAJ_NAME="5_rgb_${PERTURBATION_SET}"
    TASK_VIDEO_DIR="${VIDEOS_DIR}/${ENV_ID}/${PERTURBATION_SET}"

    if is_unsupported_combo "${ENV_ID}" "${PERTURBATION_SET}"; then
      echo "Skipping ${ENV_ID} / ${PERTURBATION_SET} (perturbation disabled by env)"
      continue
    fi

    existing_count=0
    if [[ -d "${TASK_VIDEO_DIR}" ]]; then
      existing_count=$(find "${TASK_VIDEO_DIR}" -maxdepth 1 -name '*.mp4' | wc -l)
    fi
    if [[ "${existing_count}" -ge "${NUM_TRAJ}" ]]; then
      echo "Skipping ${ENV_ID} / ${PERTURBATION_SET} (${existing_count} videos already present)"
      continue
    fi

    echo "========================================"
    echo "Collecting ${NUM_TRAJ} demos for ${ENV_ID} / ${PERTURBATION_SET}"
    echo "========================================"

    python mani_skill/examples/motionplanning/panda/run.py \
      --env-id "${ENV_ID}" \
      --obs-mode rgb \
      --num-traj "${NUM_TRAJ}" \
      --only-count-success \
      --perturbation-set "${PERTURBATION_SET}" \
      --num-procs 1 \
      --traj-name "${TRAJ_NAME}" \
      --reward-mode none \
      --random-seed

    H5_PATH="demos/${ENV_ID}/motionplanning/${TRAJ_NAME}.h5"
    if [[ ! -f "${H5_PATH}" ]]; then
      echo "ERROR: expected trajectory file missing: ${H5_PATH}" >&2
      exit 1
    fi

    echo "Extracting images/videos from ${H5_PATH}"
    python scripts/extract_h5_images.py \
      --h5-file "${H5_PATH}" \
      --save-video

    COMBINED_DIR="demos/${ENV_ID}/motionplanning/combined"
    mkdir -p "${TASK_VIDEO_DIR}"

    if [[ ! -d "${COMBINED_DIR}" ]]; then
      echo "ERROR: combined video dir missing: ${COMBINED_DIR}" >&2
      exit 1
    fi

    echo "Moving combined videos to ${TASK_VIDEO_DIR}"
    shopt -s nullglob
    for video in "${COMBINED_DIR}"/*.mp4; do
      mv "${video}" "${TASK_VIDEO_DIR}/"
    done
    shopt -u nullglob

    echo "Done: ${ENV_ID} / ${PERTURBATION_SET}"
  done
done

echo "All videos saved under ${VIDEOS_DIR}/"

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --job-name=jepa-wm-v3-batch-zsp
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/jepa-wm-v3-batch-zsp_%j.out
#SBATCH --error=<your-log-directory-setting>/jepa-wm-v3-batch-zsp_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

# Match evaluate_jepa_wm_v3_latent_prediction_seq_gczsp.sbatch's checkpoint and sampling.
# Use the original Singularity environment and read-only camera/log overlays.
# SCENE_BATCH_SIZE defaults to 64. ROLLOUT_CHUNK_SIZE caps total scene/candidate pairs.
# Stage-one extended comfort uses adjacent scenes, including supporting predecessors outside NUM_SAMPLES.
# Report pose errors and GT Hit@1/Hit@5 at 0.5s, 1s, 2s, and 4s with fractional tie credit.
# Existing trajectories and EPDMS stay anchor-only; averages include requested scenes only.
export PROJECT_ROOT="${PROJECT_ROOT:-<your-project-root-setting>}"
export CHECKPOINT="${CHECKPOINT:-<your-checkpoint-setting>}"
export CACHE_PATH="${CACHE_PATH:-$PROJECT_ROOT/navsim_experiments/metric_cache}"

cd "$PROJECT_ROOT"

export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$PROJECT_ROOT/download/maps}"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export JEPAWM_HOME="${JEPAWM_HOME:-$PROJECT_ROOT/download/deps}"
export JEPAWM_OSSCKPT="${JEPAWM_OSSCKPT:-$PROJECT_ROOT/download/deps/pretrained_models}"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$PROJECT_ROOT/navsim_experiments}"
export HYDRA_FULL_ERROR=1

SCENE_BATCH_SIZE="${SCENE_BATCH_SIZE:-16}"
NUM_CANDIDATE_TRAJ="${NUM_CANDIDATE_TRAJ:-1024}"
ROLLOUT_CHUNK_SIZE="${ROLLOUT_CHUNK_SIZE:-$((SCENE_BATCH_SIZE * NUM_CANDIDATE_TRAJ))}"

singularity exec --nv --fakeroot \
  --overlay "$PROJECT_ROOT/navsim.ext3:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/camera_test.sqf:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/navsim_logs_test.sqf:ro" \
  "<your-container-image-setting>" \
  /bin/bash -lc '
    set -euo pipefail
    source /ext3/env.sh
    conda activate ad-e2e-jepa
    cd "$PROJECT_ROOT"
    exec "$@"
  ' bash \
  python -u scripts/evaluation_jepa/evaluate_jepa_wm_compute_trajectory_batch.py \
  --agent_config "${AGENT_CONFIG:-jepa_wm_agent_v3}" \
  --checkpoint "$CHECKPOINT" \
  --num_samples "${NUM_SAMPLES:-12146}" \
  --split navtest \
  --navsim_log_path "${NAVSIM_LOG_PATH:-$OPENSCENE_DATA_ROOT/navsim_logs/test}" \
  --original_sensor_path "${ORIGINAL_SENSOR_PATH:-/}" \
  --cache_path "$CACHE_PATH" \
  --interval "${INTERVAL:-1}" \
  --scene_batch_size "$SCENE_BATCH_SIZE" \
  --num_candidate_traj "$NUM_CANDIDATE_TRAJ" \
  --rollout_chunk_size "$ROLLOUT_CHUNK_SIZE" \
  --extra_tag 0914_1213_v3_pose_horizons_gt_hits \
  --output_path "${OUTPUT_PATH:-$PROJECT_ROOT/scripts/evaluation_jepa/evaluation_jepa_results/jepa_wm_v3_stage1_epdms_0914_1213_v3_pose_horizons_gt_hits_${SLURM_JOB_ID:-manual}.csv}" \
  "$@"

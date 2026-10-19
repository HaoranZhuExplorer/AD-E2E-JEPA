#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --job-name=jepa-wm-zero-shot-planning
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/jepa-wm-zsp_%j.out
#SBATCH --error=<your-log-directory-setting>/jepa-wm-zsp_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-<your-project-root-setting>}"
export CHECKPOINT="<your-checkpoint-setting>"
export VOCABULARY_PATH="$PROJECT_ROOT/planning_vocabulary/8192.npy"
export CACHE_PATH="$PROJECT_ROOT/navsim_experiments/metric_cache"

cd "$PROJECT_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -f "$VOCABULARY_PATH" ]]; then
  echo "Planning vocabulary does not exist: $VOCABULARY_PATH" >&2
  exit 1
fi

if [[ ! -d "$CACHE_PATH" ]]; then
  echo "Metric cache directory does not exist: $CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/navsim_experiments" "$PROJECT_ROOT/scripts/evaluation_jepa/evaluation_jepa_results"

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

    export OPENSCENE_DATA_ROOT=""
    export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/download/maps"
    export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
    export JEPAWM_HOME="$PROJECT_ROOT/download/deps"
    export JEPAWM_OSSCKPT="$PROJECT_ROOT/download/deps/pretrained_models"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    export NAVSIM_EXP_ROOT="$PROJECT_ROOT/navsim_experiments"
    export HYDRA_FULL_ERROR=1

    python -u scripts/evaluation_jepa/evaluate_jepa_wm_latent_prediction_seq_gczsp.py \
      --checkpoint "$CHECKPOINT" \
      --num_samples 100 \
      --planning_warmup_runs 1 \
      --split navtest \
      --original_sensor_path / \
      --num_candidate_traj 256 \
      --num_chunks 256 \
      --extra_tag vanilla_with-planning-time
  '

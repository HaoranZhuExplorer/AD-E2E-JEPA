#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --job-name=vit-pf-proj-v2-pdm-s1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/vit-pf-proj-v2-pdm-s1_%j.out
#SBATCH --error=<your-log-directory-setting>/vit-pf-proj-v2-pdm-s1_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

export PROJECT_ROOT="<your-project-root-setting>"
export CHECKPOINT="<your-agent-checkpoint-setting>"
export CACHE_PATH="$PROJECT_ROOT/navsim_experiments/metric_cache"

cd "$PROJECT_ROOT"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 1
fi

if [[ ! -d "$CACHE_PATH" ]]; then
  echo "Metric cache directory does not exist: $CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/navsim_experiments"

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

    test -d /navsim_logs/test
    test -f "$NUPLAN_MAPS_ROOT/nuplan-maps-v1.0.json"

    python navsim/planning/script/run_pdm_score_one_stage.py \
      train_test_split=navtest \
      agent=vit_perception_free_agent_v2 \
      worker=single_machine_thread_pool \
      worker.max_workers=16 \
      worker.use_process_pool=true \
      agent.config.model.with_proj=True \
      +agent.checkpoint_path="\"${CHECKPOINT}\"" \
      experiment_name=vit_perception_free_projector_v2_agent_stage1 \
      original_sensor_path=/ \
      metric_cache_path="$CACHE_PATH"
  '

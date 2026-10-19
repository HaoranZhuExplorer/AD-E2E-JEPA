#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=navsim-cache
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/navsim-cache_%j.out
#SBATCH --error=<your-log-directory-setting>/navsim-cache_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

export PROJECT_ROOT="<your-project-root-setting>"
cd "$PROJECT_ROOT"

export OPENSCENE_DATA_ROOT=""
export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/download/maps"
export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
export JEPAWM_HOME="$PROJECT_ROOT/download/deps"
export JEPAWM_OSSCKPT="$JEPAWM_HOME/pretrained_models"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/navsim_experiments"
mkdir -p "$NAVSIM_EXP_ROOT"

unset SSL_CERT_FILE
unset SSL_CERT_DIR

# Caching settings exported to the container.
export TRAIN_TEST_SPLIT=navtrain
export EXPERIMENT_NAME="cache_navsim_${TRAIN_TEST_SPLIT}"
export CACHE_PATH="$PROJECT_ROOT/download/cache/cache_navsim_${TRAIN_TEST_SPLIT}_4hist"

singularity exec --nv --fakeroot \
  --overlay "$PROJECT_ROOT/navsim.ext3:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/camera_all.sqf:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/navsim_logs.sqf:ro" \
  "<your-container-image-setting>" \
  /bin/bash -lc '
    set -euo pipefail
    source /ext3/env.sh
    conda activate ad-e2e-jepa

    cd "$PROJECT_ROOT"
    export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/download/maps"
    export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
    export JEPAWM_HOME="$PROJECT_ROOT/download/deps"
    export JEPAWM_OSSCKPT="$JEPAWM_HOME/pretrained_models"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"

    # SSL: use the certifi bundle from this environment.
    PEM=$(python -c "import certifi; print(certifi.where())")
    export SSL_CERT_FILE="$PEM"
    export REQUESTS_CA_BUNDLE="$PEM"
    export CURL_CA_BUNDLE="$PEM"
    export SSL_CERT_DIR=/etc/ssl/certs

    # Optional local W&B storage; caching uses offline mode.
    export WANDB_DIR="<your-wandb-runs-directory-setting>"
    export WANDB_CACHE_DIR="<your-wandb-cache-directory-setting>"
    mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
    unset WANDB_VERIFY_SSL
    export WANDB_MODE=offline

    python -c "import navsim; print(\"navsim OK:\", navsim.__file__)"

    python navsim/planning/script/run_dataset_caching.py \
      agent=vit_perception_free_agent_v2 \
      train_test_split="$TRAIN_TEST_SPLIT" \
      experiment_name="$EXPERIMENT_NAME" \
      original_sensor_path=/ \
      cache_path="$CACHE_PATH" \
      force_cache_computation=True \
      worker.threads_per_node=8
  '

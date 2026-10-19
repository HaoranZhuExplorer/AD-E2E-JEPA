#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=200GB
#SBATCH --job-name=jepa-wm-v3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/jepa-wm-v3_%j.out
#SBATCH --error=<your-log-directory-setting>/jepa-wm-v3_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

export PROJECT_ROOT="<your-project-root-setting>"
CACHE_PATH="$PROJECT_ROOT/download/cache/cache_navtrain_4hist8future"

cd "$PROJECT_ROOT"

if [[ ! -d "$CACHE_PATH" ]]; then
  echo "Cache directory does not exist: $CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/navsim_experiments"

singularity exec --nv --fakeroot \
  --overlay "$PROJECT_ROOT/navsim.ext3:ro" \
  "<your-container-image-setting>" \
  /bin/bash -lc '
    set -euo pipefail

    source /ext3/env.sh
    conda activate ad-e2e-jepa

    CACHE_PATH="$PROJECT_ROOT/download/cache/cache_navtrain_4hist8future"

    cd "$PROJECT_ROOT"

    export OPENSCENE_DATA_ROOT=""
    export NUPLAN_MAPS_ROOT="$PROJECT_ROOT/download/maps"
    export NAVSIM_DEVKIT_ROOT="$PROJECT_ROOT"
    export JEPAWM_HOME="$PROJECT_ROOT/download/deps"
    export JEPAWM_OSSCKPT="$PROJECT_ROOT/download/deps/pretrained_models"
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    export NAVSIM_EXP_ROOT="$PROJECT_ROOT/navsim_experiments"
    export HYDRA_FULL_ERROR=1

    PEM=$(python -c "import certifi; print(certifi.where())")
    export SSL_CERT_FILE="$PEM"
    export REQUESTS_CA_BUNDLE="$PEM"
    export CURL_CA_BUNDLE="$PEM"
    export SSL_CERT_DIR=/etc/ssl/certs

    export WANDB_DIR="<your-wandb-runs-directory-setting>"
    export WANDB_CACHE_DIR="<your-wandb-cache-directory-setting>"
    mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
    unset WANDB_VERIFY_SSL
    export WANDB_MODE=online

    python -c "import navsim; print(\"navsim OK:\", navsim.__file__)"
    nvidia-smi

    EXPERIMENT_NAME="$(date +%m%d_%H%M)_jepa_wm_v3_navtrain_gpu-1_bs-128_8step-rollout_with_proj_patch-pred-only_patch_sigreg-0.09"

    python navsim/planning/script/run_training.py \
      agent=jepa_wm_agent_v3 \
      experiment_name="$EXPERIMENT_NAME" \
      train_test_split=navtrain \
      use_cache_without_dataset=True \
      force_cache_computation=False \
      cache_path="$CACHE_PATH" \
      train_test_split.scene_filter.frame_interval=1 \
      trainer.params.devices=1 \
      dataloader.params.batch_size=128 \
      agent.lr=0.0001 \
      agent.config.model.sigreg_patch_weight=0.09 \
      trainer.params.gradient_clip_val=1.0 \
      trainer.params.gradient_clip_algorithm=norm \
      trainer.params.max_epochs=30
  '

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=26:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100"
#SBATCH --mem=200GB
#SBATCH --job-name=vit-pf-proj-v2-rand-proj
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/vit-pf-proj-v2-rand-proj_%j.out
#SBATCH --error=<your-log-directory-setting>/vit-pf-proj-v2-rand-proj_%j.err
#SBATCH --account=<your-slurm-account-setting>

# Replace all <your-...-setting> placeholders before submission.
# The log directory must already exist when submitting with sbatch.
# This launcher assumes a Conda environment named ad-e2e-jepa, stored in
# $PROJECT_ROOT/navsim.ext3 (a Singularity ext3 overlay image).
# Inside the container, /ext3/env.sh must initialize Conda.
# Adjust the overlay path, setup script, and environment name for your system.
set -euo pipefail

export PROJECT_ROOT="<your-project-root-setting>"
CACHE_PATH="$PROJECT_ROOT/download/cache/cache_navsim_navtrain_4hist"

cd "$PROJECT_ROOT"

if [[ ! -d "$CACHE_PATH" ]]; then
  echo "Cache directory does not exist: $CACHE_PATH" >&2
  exit 1
fi

mkdir -p "$PROJECT_ROOT/navsim_experiments"

srun --ntasks=2 --ntasks-per-node=2 --kill-on-bad-exit=1 \
  singularity exec --nv --fakeroot \
  --overlay "$PROJECT_ROOT/navsim.ext3:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/camera_all.sqf:ro" \
  --overlay "$PROJECT_ROOT/download/sqf/navsim_logs.sqf:ro" \
  "<your-container-image-setting>" \
  /bin/bash -lc '
    set -euo pipefail

    source /ext3/env.sh
    conda activate ad-e2e-jepa

    CACHE_PATH="$PROJECT_ROOT/download/cache/cache_navsim_navtrain_4hist"

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

    EXPERIMENT_NAME="$(date +%m%d_%H%M)_vit_perception_free_projector_v2_full_finetuning_rand_proj_navtrain_gpu-2_bs-16-per-gpu"

    python navsim/planning/script/run_training.py \
      agent=vit_perception_free_agent_v2 \
      experiment_name="$EXPERIMENT_NAME" \
      train_test_split=navtrain \
      use_cache_without_dataset=True \
      force_cache_computation=False \
      cache_path="$CACHE_PATH" \
      train_test_split.scene_filter.frame_interval=1 \
      trainer.params.devices=2 \
      trainer.params.strategy=ddp \
      trainer.params.precision=bf16 \
      dataloader.params.batch_size=16 \
      dataloader.params.num_workers=16 \
      dataloader.params.prefetch_factor=4 \
      agent.lr=0.0001 \
      agent.config.model.visual_encoder.pretrained=true \
      agent.config.model.visual_encoder.projector_checkpoint_path=null \
      agent.config.model.unfreeze_encoder=True \
      agent.config.model.with_proj=True \
      agent.config.model.unfreeze_projector=True \
      trainer.params.gradient_clip_val=0.0 \
      trainer.params.gradient_clip_algorithm=norm \
      trainer.params.max_epochs=40
  '

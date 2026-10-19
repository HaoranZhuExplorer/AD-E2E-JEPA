#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=navtest-metric-cache
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-setting>
#SBATCH --output=<your-log-directory-setting>/navtest-metric-cache_%j.out
#SBATCH --error=<your-log-directory-setting>/navtest-metric-cache_%j.err
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
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/navsim_experiments"
mkdir -p "$NAVSIM_EXP_ROOT"

export TRAIN_TEST_SPLIT=navtest
export CACHE_PATH="$NAVSIM_EXP_ROOT/metric_cache"

singularity exec --fakeroot \
  --overlay "$PROJECT_ROOT/navsim.ext3:ro" \
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
    export PYTHONPATH="$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
    export NAVSIM_EXP_ROOT="$PROJECT_ROOT/navsim_experiments"

    test -d /navsim_logs/test
    test -f "$NUPLAN_MAPS_ROOT/nuplan-maps-v1.0.json"

    python navsim/planning/script/run_metric_caching.py \
      train_test_split="$TRAIN_TEST_SPLIT" \
      metric_cache_path="$CACHE_PATH" \
      force_feature_computation=False \
      worker=single_machine_thread_pool \
      worker.max_workers="${SLURM_CPUS_PER_TASK:-16}" \
      worker.use_process_pool=true
  '

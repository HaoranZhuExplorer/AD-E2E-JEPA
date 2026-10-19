# Training

Run these commands from the repository root on a Slurm cluster. Complete the [installation guide](INSTALL.md) first. Before submitting a job, replace the `<your-...-setting>` placeholders in its `.sbatch` script, adjust the cluster settings, and create the configured log directory.

## 1) Create a cache

For the `navtrain` split:

```bash
sbatch scripts/training/run_dataset_caching.sbatch
```

For the `trainval` split:

```bash
sbatch scripts/training/run_dataset_caching_trainval.sbatch
```

## 2) Train AD-E2E-JEPA on `navtrain`

AD-E2E-JEPA without the eight-step rollout (also called JEPA-WM v2):

```bash
sbatch scripts/training/run_jepa_wm_v2_training_with_cache.sbatch
```

AD-E2E-JEPA with the eight-step rollout (also called JEPA-WM v3):

```bash
sbatch scripts/training/run_jepa_wm_v3_training_with_cache.sbatch
```

## 3) Other training setups

Implementations of the core JEPA-WM, DINO-WM, and Le-WM agents code are available in `navsim/agents/`. We are testing their Slurm training launchers for reproducibility with 4 GPUs and plan to include them in the next release. We also plan to provide an AD-E2E-JEPA launcher for training on `trainval` in that release. You could adapt the existing sbatch scripts for those methods accordingly if needed.
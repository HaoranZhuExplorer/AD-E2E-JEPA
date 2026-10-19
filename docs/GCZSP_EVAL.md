# Goal-Conditioned Zero-Shot Planning Evaluation

Run these commands from the repository root on a Slurm cluster. Complete the [installation guide](INSTALL.md) first. Before submitting a job, replace the `<your-...-setting>` placeholders in its `.sbatch` script, adjust the cluster settings, and create the configured log directory.

## 1) Create a metric cache

Create the `navtest` metric cache at `$PROJECT_ROOT/navsim_experiments/metric_cache`:

```bash
sbatch scripts/evaluation/run_metric_caching.sbatch
```

Wait for caching to complete before submitting evaluation. Metric-cache metadata stores absolute paths; update its index or regenerate the cache if you move it.

## 2) Evaluate a sampled subset of `navtest`

Set `<your-checkpoint-setting>` in the selected launcher to a trained checkpoint matching the method and agent configuration. See the [training guide](TRAIN.md) for AD-E2E-JEPA training. All five methods use the fixed trajectory vocabulary at `planning_vocabulary/8192.npy`.

These launchers evaluate up to 100 scenes, selecting every 100th scene by default.

AD-E2E-JEPA without the eight-step rollout (also called JEPA-WM v2):

```bash
sbatch scripts/evaluation_jepa/evaluate_jepa_wm_v2_latent_prediction_seq_gczsp.sbatch
```

AD-E2E-JEPA with the eight-step rollout (also called JEPA-WM v3):

```bash
sbatch scripts/evaluation_jepa/evaluate_jepa_wm_v3_latent_prediction_seq_gczsp.sbatch
```

Original JEPA-WM:

```bash
sbatch scripts/evaluation_jepa/evaluate_jepa_wm_latent_prediction_seq_gczsp.sbatch
```

DINO-WM:

```bash
sbatch scripts/evaluation_jepa/evaluate_dino_wm_latent_prediction_seq_gczsp.sbatch
```

Le-WM:

```bash
sbatch scripts/evaluation_jepa/evaluate_le_wm_latent_prediction_seq_gczsp.sbatch
```

## 3) Evaluate the full `navtest` split

Batch evaluation launchers are provided for AD-E2E-JEPA (JEPA-WM v2/v3) and Le-WM. Set `<your-checkpoint-setting>` to a matching checkpoint before submitting. Each launcher defaults to `NUM_SAMPLES=12146` and `INTERVAL=1`, evaluating every eligible cached scene up to this limit. Ensure the metric cache covers the full split.

AD-E2E-JEPA without the eight-step rollout (also called JEPA-WM v2):

```bash
sbatch scripts/evaluation_jepa/evaluate_jepa_wm_v2_compute_trajectory_batch.sbatch
```

AD-E2E-JEPA with the eight-step rollout (also called JEPA-WM v3), using 256 candidate trajectories:

```bash
sbatch scripts/evaluation_jepa/evaluate_jepa_wm_v3_compute_trajectory_batch_256.sbatch
```

For JEPA-WM v3, replace `_256.sbatch` with `_512.sbatch`, `_1024.sbatch`, `_2048.sbatch`, `_4096.sbatch`, or `_8192.sbatch` to use the corresponding number of candidates.

Le-WM:

```bash
sbatch scripts/evaluation_jepa/evaluate_le_wm_compute_trajectory_batch.sbatch
```

The batch launchers support environment overrides for `NUM_SAMPLES`, `INTERVAL`, `SCENE_BATCH_SIZE`, `NUM_CANDIDATE_TRAJ`, and `ROLLOUT_CHUNK_SIZE`. `SCENE_BATCH_SIZE` controls scenes per batch; `ROLLOUT_CHUNK_SIZE` caps the total scene/candidate pairs per model rollout.

The batch evaluator aggregates stage-one EPDMS, including two-frame extended comfort where adjacent scenes are available. CSV results are saved under `scripts/evaluation_jepa/evaluation_jepa_results/`; set `OUTPUT_PATH` to change the destination.

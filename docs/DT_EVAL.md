# Downstream Imitation Learning Evaluation

Run these commands from the repository root on a Slurm cluster. Complete the [installation guide](INSTALL.md) first. Before submitting a job, replace the `<your-...-setting>` placeholders in its `.sbatch` script, adjust the cluster settings, and create the configured log directory.

## 1) Create a metric cache

Create the `navtest` metric cache at `$PROJECT_ROOT/navsim_experiments/metric_cache`:

```bash
sbatch scripts/evaluation/run_metric_caching.sbatch
```

## 2) Evaluate the downstream agent on `navtest`

After metric caching completes, set `<your-agent-checkpoint-setting>` in the evaluation launcher to a trained `vit_perception_free_agent_v2` checkpoint from [downstream training](TRAIN_DOWNSTREAM.md).

```bash
sbatch scripts/evaluation/run_vit_perception_free_projector_v2_pdm_score_evaluation_stage1.sbatch
```

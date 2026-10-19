# Training Downstream Imitation Learning Agent

Run these commands from the repository root on a Slurm cluster. Complete the [installation guide](INSTALL.md) first. Before submitting a job, replace the `<your-...-setting>` placeholders in its `.sbatch` script, adjust the cluster settings, and create the configured log directory.

## 1) Create a downstream cache

Create the `navtrain` cache used by both training launchers:

```bash
sbatch scripts/training/run_dataset_vit_perception_free_v2_caching.sbatch
```

## 2) Train the downstream agent on `navtrain`

Both launchers train `vit_perception_free_agent_v2`, the ViT-based imitation learning agent used in the paper.

Randomly initialized projector:

```bash
sbatch scripts/training/run_vit_perception_free_projector_v2_training_full_finetuning_rand_proj_with_cache.sbatch
```

Pretrained projector: set `<your-projector-checkpoint-setting>` in the launcher to a JEPA-WM checkpoint containing trained visual projector weights.

```bash
sbatch scripts/training/run_vit_perception_free_projector_v2_training_full_finetuning_pretrained_proj_with_cache.sbatch
```

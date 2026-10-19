# Installation

AD-E2E-JEPA uses [NAVSIM](https://github.com/autonomousvision/navsim) for training and evaluation and [DINOv3](https://github.com/facebookresearch/dinov3) as its visual encoder. This repository requires **Python 3.10**.

## Dataset and Environment

Follow the [NAVSIM installation guide](https://github.com/autonomousvision/navsim/blob/main/docs/install.md) for dataset downloads, directory structure, and environment setup. Use this repository in place of the upstream NAVSIM checkout and create the environment using the `environment.yml` and `requirements.txt` files provided here.

The supplied `.sbatch` launchers assume a Conda environment named `ad-e2e-jepa`. Since `environment.yml` defaults to `navsim`, use `--name ad-e2e-jepa` when creating the environment or update the launchers' `conda activate` command. In the supplied Singularity setup, `$PROJECT_ROOT/navsim.ext3` is the overlay containing the environment, and `/ext3/env.sh` initializes Conda inside the container. Adjust these paths for your system.

Run the installation commands and the following exports from this repository's root directory. Replace the placeholder paths with your local dataset and map directories:

```bash
export NAVSIM_DEVKIT_ROOT="$(pwd)"
export OPENSCENE_DATA_ROOT="/path/to/navsim/data"
export NUPLAN_MAPS_ROOT="/path/to/maps"
export NAVSIM_EXP_ROOT="$NAVSIM_DEVKIT_ROOT/navsim_experiments"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

`OPENSCENE_DATA_ROOT` should contain `navsim_logs/<split>` and `sensor_blobs/<split>`. `NUPLAN_MAPS_ROOT` should point to the map directory, and `NAVSIM_EXP_ROOT` is the output directory for experiment results and caches.

## Prepare DINOv3

Clone the DINOv3 source repository into the dependency directory used by the training and evaluation scripts:

```bash
export JEPAWM_HOME="$NAVSIM_DEVKIT_ROOT/download/deps"
export JEPAWM_OSSCKPT="$JEPAWM_HOME/pretrained_models"

mkdir -p "$JEPAWM_HOME" "$JEPAWM_OSSCKPT"
git clone https://github.com/facebookresearch/dinov3.git "$JEPAWM_HOME/dinov3"
```

Follow the [DINOv3 pretrained-weight instructions](https://github.com/facebookresearch/dinov3#pretrained-models) to obtain the ViT-L/16 checkpoint pretrained on LVD-1689M. Place it at:

```text
download/deps/pretrained_models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Both DINO-WM and JEPA-WM load this checkpoint directly from `JEPAWM_OSSCKPT`. Keep `JEPAWM_HOME` and `JEPAWM_OSSCKPT` set when running training or evaluation, or set them in your job launcher.

Before using the supplied `.sbatch` launchers, replace any `<your-...-setting>` placeholders, update cluster-specific paths and resource settings, and create the configured log directory. The launchers may override the environment variables above.

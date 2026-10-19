import logging
from pathlib import Path
from typing import Tuple

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_scene_filters(cfg: DictConfig) -> Tuple[SceneFilter, SceneFilter]:
    """Apply the selected scene split within the training and validation log partitions."""
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    return train_scene_filter, val_scene_filter


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter, val_scene_filter = build_scene_filters(cfg)

    data_path = Path(cfg.navsim_log_path)
    original_sensor_path = Path(cfg.original_sensor_path)

    train_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        original_sensor_path=original_sensor_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    if cfg.agent.get("checkpoint_path") is not None:
        logger.info(f"Loading agent checkpoint weights from {cfg.agent.checkpoint_path}")
        agent.initialize()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        if hasattr(agent, "build_cache_datasets"):
            train_data, val_data = agent.build_cache_datasets(cfg)
        else:
            train_scene_filter, val_scene_filter = build_scene_filters(cfg)
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=train_scene_filter.log_names,
                tokens=train_scene_filter.tokens,
            )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=val_scene_filter.log_names,
                tokens=val_scene_filter.tokens,
            )
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)
    
    logger.info(f"Num training samples: {len(train_data)}")
    logger.info(f"Num validation samples: {len(val_data)}")

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))


    logger.info("Building Trainer")
    
    # Calculate warmup iters for 1 epoch
    accumulate_grad_batches = cfg.trainer.params.get("accumulate_grad_batches", 1)
    limit_train_batches = cfg.trainer.params.get("limit_train_batches", 1.0)
    
    # If limit_train_batches is an int, use it directly (if less than len)
    # If float, it's a fraction of the dataset.
    num_devices = max(1, cfg.trainer.params.get("devices", 1))
    total_batches = len(train_dataloader) // num_devices
    if isinstance(limit_train_batches, int):
        total_batches = min(limit_train_batches, total_batches)
    elif isinstance(limit_train_batches, float):
        total_batches = int(total_batches * limit_train_batches)
        
    steps_per_epoch = total_batches // accumulate_grad_batches
    logger.info(f"Setting warmup_iters to {steps_per_epoch} (1 epoch)")
    if hasattr(agent, "set_warmup_iters"):
        agent.set_warmup_iters(steps_per_epoch)
    if hasattr(agent, "set_max_iters"):
        agent.set_max_iters(steps_per_epoch * cfg.trainer.params.get("max_epochs"))

    logger.info("Building Trainer")
    
    # Initialize WandB Logger
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(
        project=cfg.get("project", "navsim_training"),
        name=cfg.get("experiment_name", "dino_wm_experiment"),
        save_dir=cfg.output_dir,
        log_model=False
    )

    from pytorch_lightning.callbacks import LearningRateMonitor, Callback
    
    class GradNormCallback(Callback):
        """
        Logs the gradient norm.
        """
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            grad_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            pl_module.log("grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    grad_monitor = GradNormCallback()
    
    callbacks = agent.get_training_callbacks()
    callbacks.append(lr_monitor)
    callbacks.append(grad_monitor)

    trainer_params = dict(cfg.trainer.params)
    if "track_grad_norm" in trainer_params:
        trainer_params.pop("track_grad_norm")

    trainer = pl.Trainer(
        **trainer_params, 
        callbacks=callbacks,
        logger=wandb_logger
    )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=cfg.get("resume_checkpoint_path"),
    )



    # print("debug lightning_module", lightning_module)
    # exit()
    # for i, x in enumerate(train_dataloader):
    #     print(type(x), len(x), type(x[0]), x[0].keys(), type(x[1]), x[1].keys())
    #     exit()



if __name__ == "__main__":
    main()

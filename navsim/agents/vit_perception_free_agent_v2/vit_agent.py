from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.vit_perception_free_agent_v2.vit_config import (
    ViTPerceptionFreeConfig,
)
from navsim.agents.vit_perception_free_agent_v2.vit_features import (
    ViTPerceptionFreeFeatureBuilder,
    ViTPerceptionFreeTrajectoryTargetBuilder,
)
from navsim.agents.vit_perception_free_agent_v2.vit_model import (
    ViTPerceptionFreeModel,
)
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class ViTPerceptionFreeAgent(AbstractAgent):
    """ViT perception-free trajectory agent."""

    def __init__(
        self,
        config: ViTPerceptionFreeConfig,
        lr: float = 1e-4,
        encoder_projector_lr: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(
            time_horizon=4, interval_length=0.5
        ),
    ):
        super().__init__(trajectory_sampling)
        self._config = config
        self._lr = lr
        self._encoder_projector_lr = encoder_projector_lr
        self._checkpoint_path = checkpoint_path
        self._model = ViTPerceptionFreeModel(self._trajectory_sampling, config)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path is None:
            return
        checkpoint = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict: Dict[str, Any] = checkpoint["state_dict"]
        self.load_state_dict(
            {key.replace("agent.", ""): value for key, value in state_dict.items()}
        )

    def get_sensor_config(self) -> SensorConfig:
        num_history_frames = self._config.model["num_history_frames"]
        return SensorConfig(
            cam_f0=list(range(num_history_frames)),
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ViTPerceptionFreeFeatureBuilder(self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            ViTPerceptionFreeTrajectoryTargetBuilder(
                trajectory_sampling=self._trajectory_sampling,
            )
        ]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pred, gt = predictions["trajectory"], targets["trajectory"]
        return l1_length_normalized_loss(pred, gt, alpha=5.0)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if self._encoder_projector_lr is None:
            return torch.optim.Adam(trainable_parameters, lr=self._lr)

        encoder_projector_parameters = [
            parameter
            for parameter in self._model.backbone.parameters()
            if parameter.requires_grad
        ]
        if self._model.visual_projector is not None:
            encoder_projector_parameters.extend(
                parameter
                for parameter in self._model.visual_projector.parameters()
                if parameter.requires_grad
            )

        encoder_projector_parameter_ids = {
            id(parameter) for parameter in encoder_projector_parameters
        }
        decoder_parameters = [
            parameter
            for parameter in trainable_parameters
            if id(parameter) not in encoder_projector_parameter_ids
        ]

        parameter_groups = []
        if encoder_projector_parameters:
            parameter_groups.append(
                {
                    "params": encoder_projector_parameters,
                    "lr": self._encoder_projector_lr,
                }
            )
        if decoder_parameters:
            parameter_groups.append({"params": decoder_parameters, "lr": self._lr})

        return torch.optim.Adam(parameter_groups, lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="epoch={epoch}-val_loss={val/loss:.4f}",
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                monitor=None,
                save_top_k=-1,
                every_n_epochs=10,
                filename="periodic-{epoch}",
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
            ),
        ]

def l1_length_normalized_loss(pred, gt, alpha=1.0, eps=1e-6):
    """
    pred, gt: [B, N, 3]  (x, y, yaw)
    Normalizes L1 loss by GT trajectory length.
    """
    # Mean L1 per sample
    per_sample_l1 = (pred - gt).abs().mean(dim=[1, 2])  # [B]

    # Compute GT arc length in XY
    dxy = gt[:, 1:, :2] - gt[:, :-1, :2]                # [B, N-1, 2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1) # [B]
    
    # Compute weights: inverse of (1 + length)
    w = 1.0 / (alpha + arc_len)

    # Normalize weights so avg weight ≈ 1
    w = w * (w.numel() / (w.sum() + eps))

    # Weighted mean loss
    loss = (w * per_sample_l1).mean()
    return loss

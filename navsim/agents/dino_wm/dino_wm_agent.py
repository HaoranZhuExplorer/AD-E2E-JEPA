from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.dino_wm.dino_wm_callback import DINO_WMCallback
from navsim.agents.dino_wm.dino_wm_config import DINO_WMConfig
from navsim.agents.dino_wm.dino_wm_features import DINO_WMFeatureBuilder, DINO_WMTargetBuilder
from navsim.agents.dino_wm.dino_wm_model import DINO_WMModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import AgentInput, Scene, Trajectory


def consecutive_relative_se2_actions(poses: torch.Tensor) -> torch.Tensor:
    """Convert (..., T + 1, 3) poses into (..., T, 3) consecutive SE(2) actions."""
    current_poses = poses[..., :-1, :]
    next_poses = poses[..., 1:, :]

    dx = next_poses[..., 0] - current_poses[..., 0]
    dy = next_poses[..., 1] - current_poses[..., 1]
    current_heading = current_poses[..., 2]

    cos_heading = torch.cos(current_heading)
    sin_heading = torch.sin(current_heading)
    x_relative = dx * cos_heading + dy * sin_heading
    y_relative = -dx * sin_heading + dy * cos_heading

    heading_delta = next_poses[..., 2] - current_heading
    heading_relative = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta))

    return torch.stack([x_relative, y_relative, heading_relative], dim=-1)


class DINO_WMAgent(AbstractAgent):
    """Agent interface for DINO_WM baseline."""

    def __init__(
        self,
        config: DINO_WMConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes DINO_WM agent.
        :param config: global config of DINO_WM agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling, requires_scene=True)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._dino_wm_model = DINO_WMModel(self._trajectory_sampling, config)
        self._warmup_iters = 1000 # Default
        self._max_iters = 100000
    
    def set_warmup_iters(self, warmup_iters: int) -> None:
        """
        Sets the number of warmup iterations.
        :param warmup_iters: number of warmup steps
        """
        self._warmup_iters = warmup_iters
    
    def set_max_iters(self, max_iters: int) -> None:
        """
        Sets the maximum number of training iterations.
        :param max_iters: maximum number of training iterations
        """
        self._max_iters = max_iters

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # Match the shared four-history/eight-future front-camera cache.
        history_steps = list(range(12)) 
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [DINO_WMTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [DINO_WMFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._dino_wm_model(features)
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        camera_feature = features["camera_feature"].squeeze(2) # (b, 4, 3, 256, 512)
        camera_feature_next = targets["camera_feature_next"].squeeze(2)[:, :1] # (b, 1, 3, 256, 512)
        
        visual = torch.cat([camera_feature, camera_feature_next], dim=1) # (b, 5, 3, 256, 512)
        obs = {
            "visual": visual,
        }
        poses = targets["trajectory"][:, : self._dino_wm_model.num_hist + 1, :]
        act = consecutive_relative_se2_actions(poses) # (b, 4, 3)

        z = self._dino_wm_model.encode_obs(obs)["visual"] # (b, 5, 512, visual_dim)
        z_src = z[:, : self._dino_wm_model.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self._dino_wm_model.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        z_src = self._dino_wm_model.concat_obs_act(z_src, self._dino_wm_model.encode_act(act))
        
        if self._dino_wm_model.predictor is not None:
            z_pred = self._dino_wm_model.predict(z_src)  # (b, 4, 512, visual_dim + action_dim)

            if self._dino_wm_model.concat_dim == 1:
                z_loss = self._dino_wm_model.emb_criterion(
                        z_pred[:, :, :, :-self._dino_wm_model.action_dim], 
                        z_tgt.detach()
                )
                

            else:
                raise Exception("concat_dim at other dim not supported.")

        return {"loss": z_loss, "pred_loss": z_loss}

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        opt_cfg = self._config.optimization["transition_model"]
        optimizer = torch.optim.AdamW(
            self._dino_wm_model.parameters(), lr=self._lr,
            weight_decay=float(opt_cfg["weight_decay"]),
            betas=tuple(opt_cfg["betas"]), eps=float(opt_cfg["eps"]),
        )
        
        warmup_iters = self._warmup_iters
        max_iters = self._max_iters
        
        scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_iters)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters, eta_min=self._lr/10)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [
            DINO_WMCallback(self._config),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="epoch={epoch}-val_loss={val/loss:.4f}",
                auto_insert_metric_name=False
            ),
            ModelCheckpoint(
                monitor=None, # Save based on epoch
                save_top_k=-1, # Keep every epoch checkpoint, as in JEPA-WM.
                every_n_epochs=1,
                filename="latest-{epoch}",
                auto_insert_metric_name=False,
                save_last=True,
                save_weights_only=False,
                save_on_train_epoch_end=True,
            )
        ]

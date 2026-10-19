import math
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR, LambdaLR, SequentialLR

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.le_wm.le_wm_callback import LE_WMCallback
from navsim.agents.le_wm.le_wm_config import LE_WMConfig
from navsim.agents.le_wm.le_wm_features import LE_WMFeatureBuilder, LE_WMTargetBuilder
from navsim.agents.le_wm.le_wm_model import LE_WM_Model
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import AgentInput, Scene, Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

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

class LE_WM_Agent(AbstractAgent):
    """Agent interface for LE_WM baseline."""

    def __init__(
        self,
        config: LE_WMConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes LE_WM agent.
        :param config: global config of LE_WM agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling, requires_scene=True)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._le_wm_model = LE_WM_Model(self._trajectory_sampling, config)
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
        if self._checkpoint_path is None:
            return
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cuda"))[
                "state_dict"
            ]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        state_dict = {k.replace("agent.", ""): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # Cache four history and eight future frames; training uses only the first future.
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
        return [LE_WMTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [LE_WMFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._le_wm_model(features)

    def compute_trajectory(
        self,
        agent_input: Union[AgentInput, List[AgentInput]],
        scene: Union[Scene, List[Scene]],
        *,
        num_candidate_traj: int = 256,
        chunk_size: Optional[int] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        Trajectory, List[Trajectory],
        tuple[Trajectory, Dict[str, Any]],
        tuple[List[Trajectory], List[Dict[str, Any]]],
    ]:
        
        """
        Compute trajectories using each scene's GT future as the latent goal.
        :param agent_input: Agent input(s); observations are read from scene.
        :param scene: One scene or a list, each with four history and eight future frames.
        :param num_candidate_traj: Number of vocabulary trajectories evaluated per scene.
        :param chunk_size: Maximum scene/candidate pairs per rollout; None evaluates all pairs together.
        :param return_diagnostics: Also return selected poses, anchor costs, and GT rollout costs
            at 0.5s, 1s, 2s, and 4s, in that order.
        :return: One Trajectory for a Scene, or a list in the same order as the input scenes.
            With diagnostics enabled, return (trajectory, diagnostics) or matching lists.
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            import numpy as np
            from pathlib import Path
            from PIL import Image

            single_scene = isinstance(scene, Scene)
            scenes = [scene] if single_scene else scene
            num_scenes = len(scenes)
            if num_scenes == 0:
                return ([], []) if return_diagnostics else []
            if chunk_size is not None and chunk_size < 1:
                raise ValueError("chunk_size must be positive or None.")
            imgs_tensors = []
            proprio_tensors = []
            global_ego_poses = []
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            frames_to_use = 12
            num_future_steps = frames_to_use - 4
            evaluation_steps = (1, 2, 4, num_future_steps) if return_diagnostics else (num_future_steps,)
            
            frames = [frame for current_scene in scenes for frame in current_scene.frames[:frames_to_use]]
            for frame in frames:
                f0 = frame.cameras.cam_f0.image
                status = frame.ego_status
                
                proprio_tensor = torch.concatenate([
                    torch.tensor(status.driving_command, dtype=torch.float32),
                    torch.tensor(status.ego_velocity, dtype=torch.float32),
                    torch.tensor(status.ego_acceleration, dtype=torch.float32),
                ], dim=0)
                
                image = Image.fromarray(f0[28:-28]).resize((512, 256))
                image = np.asarray(image, dtype=np.float32) / 255.0
                image = (image - mean) / std
                tensor_img = torch.from_numpy(image).permute(2, 0, 1)
                imgs_tensors.append(tensor_img)
                proprio_tensors.append(proprio_tensor)
                global_ego_poses.append(status.ego_pose)

            gt_seq = torch.stack(imgs_tensors, dim=0).reshape(num_scenes, frames_to_use, 3, 256, 512).to(device)
            gt_proprio = torch.stack(proprio_tensors, dim=0).reshape(num_scenes, frames_to_use, -1).to(device)
            global_ego_poses = np.array(global_ego_poses).reshape(num_scenes, frames_to_use, 3)

            # Encode the initial history and matching targets for the requested horizons.
            gt_z_src_list = []
            gt_z_tgt_by_step = {}
            
            for k in sorted({0} | {step - 1 for step in evaluation_steps}):
                curr_visual = gt_seq[:, k:k+5]
                curr_proprio = gt_proprio[:, k:k+5]
                curr_obs = {"visual": curr_visual, "proprio": curr_proprio}
                curr_ego_poses = global_ego_poses[:, k:k+5]
                curr_poses = torch.tensor(
                    curr_ego_poses, dtype=torch.float32, device=device
                )
                curr_actions = consecutive_relative_se2_actions(curr_poses)
                with torch.no_grad():
                    z = self._le_wm_model.encode_obs(curr_obs)
                    gt_z_src_list.append(z[:, : self._le_wm_model.num_hist, :])
                    gt_z_tgt_by_step[k + 1] = z[:, self._le_wm_model.num_pred :, :]

            # Evaluate candidates
            vocabulary_path = Path(__file__).resolve().parents[3] / "planning_vocabulary" / "8192.npy"
            clustered_trajectories = np.load(vocabulary_path)
            num_trajs_to_plot = num_candidate_traj
            if not 1 <= num_trajs_to_plot <= len(clustered_trajectories):
                raise ValueError("num_candidate_traj must be between 1 and the vocabulary size.")
            interval = len(clustered_trajectories) // num_trajs_to_plot
            sampled_trajs = clustered_trajectories[::interval][:num_trajs_to_plot]
            
            class BatchedWMUnroll:
                def __init__(self, agent, global_ego_poses, gt_proprio, device):
                    self.agent = agent
                    origins = global_ego_poses[:, 3]
                    self.origins = torch.tensor(
                        np.column_stack([origins, np.cos(origins[:, 2]), np.sin(origins[:, 2])]),
                        dtype=torch.float32, device=device,
                    )
                    self.history_poses = torch.tensor(
                        global_ego_poses[:, :4], dtype=torch.float32, device=device
                    )
                    self.gt_proprio = gt_proprio
                    
                def __call__(self, z_init, scene_indices, actions=None, act_suffix=None):
                    if act_suffix is not None:
                        actions = act_suffix
                    actions = actions.permute(1, 0, 2) # [B, 8, 3]
                    B = actions.shape[0]
                    
                    origin_x, origin_y, origin_yaw, c, s = self.origins[scene_indices].unsqueeze(1).unbind(-1)
                    x_local, y_local, yaw_local = actions[:, :, 0], actions[:, :, 1], actions[:, :, 2]
                    x_g = origin_x + x_local * c - y_local * s
                    y_g = origin_y + x_local * s + y_local * c
                    yaw_g = origin_yaw + yaw_local
                    future_poses = torch.stack([x_g, y_g, yaw_g], dim=-1) # [B, 8, 3]
                    history_poses = self.history_poses[scene_indices]
                    candidate_simulated_poses = torch.cat([history_poses, future_poses], dim=1) # [B, 12, 3]
                    
                    gt_z_src_expanded = z_init[scene_indices]
                    z_rollout = gt_z_src_expanded
                    gt_proprio_expanded = self.gt_proprio[scene_indices] # [B, 12, 8]
                    horizon_costs = []
                    
                    for k in range(8):
                        curr_proprio = gt_proprio_expanded[:, k:k+5] # [B, 5, 8]
                        curr_ego_poses = candidate_simulated_poses[:, k:k+5, :] # [B, 5, 3]
                        curr_actions = consecutive_relative_se2_actions(curr_ego_poses) # [B, 4, 3]
                        curr_proprio_prev = curr_proprio[:, :-1, :] # [B, 4, 8]

                        with torch.no_grad():
                            le_wm = self.agent._le_wm_model
                            cur_act_embed = le_wm.encode_act(curr_actions)

                            z_pred = le_wm.predict(z_rollout, cur_act_embed)
                            z_pred_next = z_pred[:, -1:]
                            z_rollout = torch.cat([z_rollout[:, 1:], z_pred_next], dim=1)
                            
                            z_p_loss = z_pred_next
                            if k + 1 in objective_fns:
                                horizon_costs.append(objective_fns[k + 1](z_p_loss, scene_indices))
                        
                    return torch.stack(horizon_costs, dim=1) # Retain scalar costs, not horizon latents.
                    
            class GCZSPEvalObjective:
                def __init__(self, gt_z_tgt_last, agent):
                    self.gt_z_tgt_last = gt_z_tgt_last
                    self.agent = agent
                    
                def __call__(self, encodings, scene_indices):
                    z_pred_last = encodings # [B, ...]
                    z_t_loss = self.gt_z_tgt_last[:, -1:].detach()
                    
                    z_t_loss = z_t_loss[scene_indices]
                    diff = (z_pred_last - z_t_loss) ** 2
                    return diff.mean(dim=list(range(1, diff.ndim)))
                    
            unroll_fn = BatchedWMUnroll(self, global_ego_poses, gt_proprio, device=device)
            objective_fns = {
                step: GCZSPEvalObjective(gt_z_tgt_by_step[step], self)
                for step in evaluation_steps
            }

            # Evaluate pre-clustered trajectories directly
            downsampled_trajs = sampled_trajs[:, 4::5, :]  # shape: [N, 8, 3]
            traj_8_pts_all = downsampled_trajs[:, :8, :]  # shape: [N, 8, 3]
            actions_tensor = torch.tensor(traj_8_pts_all, dtype=torch.float32, device=device) # [N, 8, 3]
            
            num_rollouts = num_scenes * num_trajs_to_plot
            if chunk_size is None:
                chunk_size = num_rollouts
            all_costs = []
            
            z_init = gt_z_src_list[0].detach()
            actions_tensor_T = actions_tensor.transpose(0, 1).detach() # [8, N, 3]
            
            # Flatten [scene, candidate] in scene-major order, including partial chunks.
            for i in range(0, num_rollouts, chunk_size):
                rollout_indices = torch.arange(i, min(i + chunk_size, num_rollouts), device=device)
                scene_indices = rollout_indices // num_trajs_to_plot
                candidate_indices = rollout_indices % num_trajs_to_plot
                chunk_actions = actions_tensor_T[:, candidate_indices, :]
                costs = unroll_fn(z_init, scene_indices, actions=chunk_actions)
                all_costs.append(costs.detach())
                
                del costs
                del chunk_actions
            
            traj_losses = torch.cat(all_costs, dim=0).reshape(num_scenes, num_trajs_to_plot, len(evaluation_steps))
            best_traj_indices = traj_losses.argmin(dim=1).cpu().tolist()
            trajectories = [
                Trajectory(sampled_trajs[idx[-1]][4::5][:8], self._trajectory_sampling)
                for idx in best_traj_indices
            ]

            if return_diagnostics:
                # GT is evaluated separately; returned trajectories always use the 4s anchor selection.
                gt_future_local = np.stack([
                    convert_absolute_to_relative_se2_array(StateSE2(*poses[3]), poses[4:12])
                    for poses in global_ego_poses
                ])
                gt_actions_tensor_T = torch.tensor(
                    gt_future_local, dtype=torch.float32, device=device
                ).transpose(0, 1)
                all_gt_costs = []
                for i in range(0, num_scenes, chunk_size):
                    scene_indices = torch.arange(i, min(i + chunk_size, num_scenes), device=device)
                    gt_costs = unroll_fn(z_init, scene_indices, actions=gt_actions_tensor_T[:, i:i+chunk_size])
                    all_gt_costs.append(gt_costs.detach())
                gt_losses = torch.cat(all_gt_costs, dim=0).cpu().numpy()
                anchor_losses = traj_losses.cpu().numpy()
                future_indices = np.array(evaluation_steps) - 1
                diagnostics = [
                    {
                        "selected_poses": traj_8_pts_all[best_traj_indices[i], future_indices],
                        "anchor_latent_mse": anchor_losses[i],
                        "gt_latent_mse": gt_losses[i],
                    }
                    for i in range(num_scenes)
                ]
                if single_scene:
                    return trajectories[0], diagnostics[0]
                return trajectories, diagnostics

        # extract trajectory
        return trajectories[0] if single_scene else trajectories

    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        camera_feature = features["camera_feature"] # (b, num_hist, 1, 3, 256, 512)
        camera_feature_next = targets["camera_feature_next"] # (b, 8, 1, 3, 256, 512)

        # The shared cache contains eight future frames; train on the first only.
        camera_feature = camera_feature.squeeze(2) # (b, num_hist, 3, 256, 512)
        camera_feature_next = camera_feature_next.squeeze(2) # (b, 8, 3, 256, 512)
        camera_feature_next = camera_feature_next[:, :1, :, :, :] # (b, 1, 3, 256, 512)
        #print("debug camera_feature shape:", camera_feature.shape, "camera_feature_next shape:", camera_feature_next.shape)

        visual = torch.cat([camera_feature, camera_feature_next], dim=1) # (b, num_hist + 1, 3, 256, 512)
        obs = {"visual": visual}
        poses = targets["trajectory"] # (b, num_hist + 8, 3)
        # Use the same four-history-plus-one-future horizon for relative actions.
        poses = poses[:, : self._le_wm_model.num_hist + 1, :] # (b, num_hist + 1, 3)
        actions = consecutive_relative_se2_actions(poses) # (b, num_hist, 3)

        z = self._le_wm_model.encode_obs(obs)
        # z: (B, T, D), one CLS embedding per frame.
        # Match SIGReg's sample semantics: compute variance across independent
        # batch samples, then average over time.
        encoder_dim_var = (
            z.detach().float().var(dim=0, unbiased=False).mean(dim=0)
        )
        sigreg_loss = self._le_wm_model.sigreg(z.transpose(0, 1))
        act_src = actions

        z_src = z[:, : self._le_wm_model.num_hist, :]  # (B, num_hist, D)
        z_tgt = z[:, self._le_wm_model.num_pred :, :]  # (B, num_hist, D)
        
        # Extract the action features corresponding to the history segment.
        action_features_src = self._le_wm_model.encode_act(act_src)

        z_pred = self._le_wm_model.predict(
            z_src,
            action_features_src,
        )  # (B, num_hist, D)
        
        z_p_loss = z_pred
        z_t_loss = z_tgt # Upstream LeWorld keeps gradients through its training targets.
        pred_loss = torch.nn.functional.mse_loss(z_p_loss, z_t_loss)

        z_loss = (
            pred_loss
            + self._le_wm_model.sigreg_weight * sigreg_loss
        )
        
        return {
            "loss": z_loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "encoder_embedding_var_mean": encoder_dim_var.mean(),
        }

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""

        # Match JEPA-WM's effective transition-model optimizer settings.
        weight_decay = 1e-7
        betas = (0.9, 0.995)
        eps = 1e-8

        parameters = self._le_wm_model.parameters()
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self._lr, 
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
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
            LE_WMCallback(self._config),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="epoch={epoch}-val_loss={val/loss:.4f}",
                auto_insert_metric_name=False
            ),
            ModelCheckpoint(
                monitor=None, # Save based on epoch
                save_top_k=-1, # Keep every epoch checkpoint
                every_n_epochs=1,
                filename="latest-{epoch}",
                auto_insert_metric_name=False,
                save_last=True,
                save_weights_only=False, # Include optimizer and LR scheduler state for resume
                save_on_train_epoch_end=True, # Save even when validation is disabled
            )
        ]

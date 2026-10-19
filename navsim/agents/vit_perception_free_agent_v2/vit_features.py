from typing import Dict

import numpy as np
import torch
from PIL import Image
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vit_perception_free_agent_v2.vit_config import (
    ViTPerceptionFreeConfig,
)
from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)


class ViTPerceptionFreeFeatureBuilder(AbstractFeatureBuilder):
    """Build four normalized front-camera frames and their ego states."""

    def __init__(self, config: ViTPerceptionFreeConfig):
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "navsim_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input) # (num_history_frames, 1, 3, H, W)
        
        # use all past statuses as input, and concatenate them along channel dimension
        statuses = agent_input.ego_statuses
        features["status_feature"] = torch.stack(
            [
                torch.concatenate(
                    [
                        torch.tensor(status.driving_command, dtype=torch.float32),
                        torch.tensor(status.ego_velocity, dtype=torch.float32),
                        torch.tensor(status.ego_acceleration, dtype=torch.float32),
                    ],
                    dim=0,
                )
                for status in statuses
            ],
            dim=0,  # stack along time dimension
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        For prototype jepa-wm exp, we only use front camera image
        """

        history_frames = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for cameras in agent_input.cameras:
            f0 = cameras.cam_f0.image[28:-28]
            image = Image.fromarray(f0).resize((512, 256))
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = (image - mean) / std
            history_frames.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        return torch.stack(history_frames, dim=0)  # (num_history_frames, 1, 3, 256, 512)



class ViTPerceptionFreeTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Build the future trajectory for downstream driving."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
    ):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "navsim_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        trajectory = torch.tensor(
            scene.get_future_trajectory(
                num_trajectory_frames=self._trajectory_sampling.num_poses
            ).poses
        )
        return {"trajectory": trajectory}

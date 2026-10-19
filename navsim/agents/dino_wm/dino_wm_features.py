from enum import IntEnum
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap, MapObject, SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image
from shapely import affinity
from shapely.geometry import LineString, Polygon

from navsim.agents.dino_wm.dino_wm_config import DINO_WMConfig
from navsim.common.dataclasses import AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class DINO_WMFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for DINO_WM."""

    def __init__(self, config: DINO_WMConfig):
        """
        Initializes feature builder.
        :param config: global config dataclass of DINO_WM
        """
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "front4hist8future_256x512_feature"
    
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}

        features["camera_feature"] = self._get_camera_feature(agent_input) # (4, 1, 3, 256, 512)
        
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
        For prototype dino-wm exp, we only use front camera image
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

class DINO_WMTargetBuilder(AbstractTargetBuilder):
    """Output target builder for DINO_WM."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: DINO_WMConfig,
    ):
        """
        Initializes target builder.
        :param trajectory_sampling: trajectory sampling specification
        :param config: global config dataclass of DINO_WM
        """
        self._trajectory_sampling = trajectory_sampling
        self._config = config

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "front4hist8future_256x512_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        num_future_frames = 8
        trajectory_future = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=num_future_frames).poses,
            dtype=torch.float32,
        ) # (num_trajectory_frames, 3)

        trajectory_past = torch.tensor(
            scene.get_history_trajectory(num_trajectory_frames=4).poses,
            dtype=torch.float32,
        ) # (num_history_frames, 3)

        trajectory = torch.cat([trajectory_past, trajectory_future], dim=0)

        future_frames = scene.frames[4 : 4 + num_future_frames]
        camera_feature_next = self._get_camera_feature(future_frames)

        future_status_feature = torch.stack(
            [
                torch.concatenate(
                    [
                        torch.tensor(frame.ego_status.driving_command, dtype=torch.float32),
                        torch.tensor(frame.ego_status.ego_velocity, dtype=torch.float32),
                        torch.tensor(frame.ego_status.ego_acceleration, dtype=torch.float32),
                    ],
                    dim=0,
                )
                for frame in future_frames
            ],
            dim=0,
        )

        return {
            "trajectory": trajectory,
            "camera_feature_next": camera_feature_next,
            "status_feature_next": future_status_feature
        }

    def _get_camera_feature(self, frames: List[Any]) -> torch.Tensor:
        future_images = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for frame in frames:
            f0 = frame.cameras.cam_f0.image[28:-28]
            image = Image.fromarray(f0).resize((512, 256))
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = (image - mean) / std
            future_images.append(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

        return torch.stack(future_images, dim=0)  # (num_future_frames, 1, 3, 256, 512)

class BoundingBox2DIndex(IntEnum):
    """Intenum for bounding boxes in DINO_WM."""

    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls):
        valid_attributes = [
            attribute
            for attribute in dir(cls)
            if attribute.startswith("_") and not attribute.startswith("__") and not callable(getattr(cls, attribute))
        ]
        return len(valid_attributes)

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        # assumes X, Y have subsequent indices
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        # assumes X, Y, HEADING have subsequent indices
        return slice(cls._X, cls._HEADING + 1)
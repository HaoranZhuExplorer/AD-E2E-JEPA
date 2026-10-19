import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.vit_perception_free_agent_v2.vit_config import (
    ViTPerceptionFreeConfig,
)
from navsim.agents.jepa_wm.jepa_wm_projector import build_visual_projector
from navsim.common.enums import StateSE2Index
from einops import rearrange

# for importing the DINOv3 backbone from jepa-wm repo
JEPA_WM_DIR = Path(__file__).resolve().parents[1] / "jepa_wm" / "jepa-wms"
if str(JEPA_WM_DIR) not in sys.path:
    sys.path.insert(0, str(JEPA_WM_DIR))
from app.plan_common.models.dino import DinoEncoder


class _DinoV3EncoderFromScratch(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        project_root = Path(__file__).resolve().parents[3]
        dinov3_root = Path(
            os.environ.get("JEPAWM_HOME", project_root / "download" / "deps")
        ) / "dinov3"
        self.base_model = torch.hub.load(
            str(dinov3_root), name, source="local", pretrained=False
        )
        self.emb_dim = self.base_model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model.forward_features(x)["x_norm_patchtokens"]


class ViTPerceptionFreeModel(nn.Module):

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config: ViTPerceptionFreeConfig,
    ):
        super().__init__()

        model_config = config.model
        encoder_config = model_config["visual_encoder"]
        decoder_config = model_config["trajectory_decoder"]

        self._num_history_frames = int(model_config["num_history_frames"])
        self._grid_size = tuple(model_config["grid_size"])
        self._projected_grid_size = tuple(model_config["projected_grid_size"])
        self._unfreeze_encoder = bool(model_config.get("unfreeze_encoder", False))
        self._with_proj = bool(model_config.get("with_proj", False))
        self._unfreeze_projector = bool(model_config.get("unfreeze_projector", False))

        if encoder_config.get("pretrained", True):
            self.backbone = DinoEncoder(
                name=encoder_config["enc_version"],
                feature_key="x_norm_patchtokens",
                causal_enc=False,
            )
            print(f"Loading pretrained weights for {encoder_config['enc_version']}")
        else:
            self.backbone = _DinoV3EncoderFromScratch(
                name=encoder_config["enc_version"]
            )
            print(f"Training {encoder_config['enc_version']} from scratch")
        if not self._unfreeze_encoder:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.visual_projector = None
        image_dim = self.backbone.emb_dim
        if self._with_proj:
            self.visual_projector = build_visual_projector(encoder_config)
            projector_checkpoint_path = encoder_config.get("projector_checkpoint_path")
            if projector_checkpoint_path:
                checkpoint = torch.load(
                    projector_checkpoint_path,
                    map_location="cpu",
                    mmap=True,
                    weights_only=False,
                )
                state_dict = checkpoint["state_dict"]
                projector_prefix = "agent._jepa_wm_model.visual_projector."
                projector_state_dict = {
                    key.removeprefix(projector_prefix): value
                    for key, value in state_dict.items()
                    if key.startswith(projector_prefix)
                }
                if not projector_state_dict:
                    raise KeyError(
                        f"No visual projector weights with prefix '{projector_prefix}' "
                        f"found in {projector_checkpoint_path}"
                    )
                self.visual_projector.load_state_dict(projector_state_dict)
                print(f"Loading pretrained visual projector from {projector_checkpoint_path}")
            else:
                print("Training visual projector from scratch")
            image_dim = self.visual_projector.out_dim
            if not self._unfreeze_projector:
                self.visual_projector.requires_grad_(False)
                self.visual_projector.eval()

        d_model = int(decoder_config["d_model"])
        d_ffn = int(decoder_config["d_ffn"])
        num_layers = int(decoder_config["num_layers"])
        num_heads = int(decoder_config["num_heads"])
        dropout = float(decoder_config["dropout"])

        self.image_fc = nn.Linear(image_dim, d_model)
        self._status_encoding = nn.Linear(8, d_model)

        num_poses = trajectory_sampling.num_poses

        self._temporal_embedding = nn.Embedding(self._num_history_frames, d_model)
        self._spatial_embedding = nn.Embedding(
            self._projected_grid_size[0] * self._projected_grid_size[1], d_model
        )
        self._query_embedding = nn.Embedding(num_poses, d_model)
        self._transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
        )
        self._trajectory_head = TrajectoryHead(num_poses, d_ffn, d_model)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        camera_feature = features["camera_feature"] # (B, K, 1, 3, H, W)
        status_feature = features["status_feature"] # (B, K, 8)

        camera_feature = camera_feature.squeeze(2)  # (B, K, 3, H, W)
        images = rearrange(camera_feature, 'B K C H W -> (B K) C H W')
        img_feat = self.backbone(images) # (B*K, H_p*W_p, D) = (B*K, 16*32, D)

        grid_height, grid_width = self._grid_size
        img_feat = rearrange(
            img_feat,
            "(B K) (H_p W_p) D -> B K D H_p W_p",
            B=camera_feature.shape[0],
            K=camera_feature.shape[1],
            H_p=grid_height,
            W_p=grid_width,
        ) # (B, K, D, 16, 32)

        if self.visual_projector is not None:
            img_feat = rearrange(img_feat, "B K D H_p W_p -> (B K) D H_p W_p")
            img_feat = self.visual_projector(img_feat) # (B*K, 256, 4, 8)
            img_feat = rearrange(
                img_feat,
                "(B K) D H_p W_p -> B K D H_p W_p",
                B=camera_feature.shape[0],
                K=camera_feature.shape[1],
            ) # (B, K, 256, 4, 8)

        img_feat = rearrange(img_feat, "B K D H W -> B K (H W) D") # (B, K, 4*8, D)
        img_feat = self.image_fc(img_feat)  # (B, K, 4*8, d_model)
        img_feat = img_feat + self._temporal_embedding.weight[None, :, None, :]
        img_feat = img_feat + self._spatial_embedding.weight[None, None, :, :]
        img_feat = rearrange(img_feat, "B K P D -> B (K P) D") # (B, K*4*8, d_model)

        status_encoding = self._status_encoding(status_feature) # (B, K, d_model)
        status_encoding = status_encoding + self._temporal_embedding.weight[None, :, :]

        keyval = torch.cat([img_feat, status_encoding], dim=1) # (B, K*4*8 + K, d_model)

        # traj decoder
        batch_size = img_feat.shape[0]
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1) # (B, num_poses, d_model)
        query_out = self._transformer(src=keyval, tgt=query) # (B, num_poses, d_model)
        trajectory = self._trajectory_head(query_out) # {"trajectory": (B, num_poses, 3)}
        return trajectory


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}

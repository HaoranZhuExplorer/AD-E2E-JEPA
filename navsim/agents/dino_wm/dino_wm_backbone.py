"""
Implements the DINO_WM vision backbone.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.dino_wm.dino_wm_config import DINO_WMConfig


class DINO_WMBackbone(nn.Module):
    """Frozen DINOv3 image encoder with padding to complete border patches."""

    def __init__(self, config: DINO_WMConfig):

        super().__init__()
        self.config = config

        if self.config.encoder_type == 'vit-s':
            self.image_encoder = torch.hub.load("facebookresearch/dinov3", 'dinov3_vits16')
        elif self.config.encoder_type == 'vit-b':
            self.image_encoder = torch.hub.load("facebookresearch/dinov3", 'dinov3_vitb16')
        elif self.config.encoder_type == 'vit-l':
            deps_root = Path(os.environ.get("JEPAWM_HOME", Path(__file__).resolve().parents[3] / "download/deps"))
            weights_root = Path(os.environ.get("JEPAWM_OSSCKPT", deps_root / "pretrained_models"))
            weights = weights_root / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            self.image_encoder = torch.hub.load(
                str(deps_root / "dinov3"), 'dinov3_vitl16', source="local", weights=str(weights)
            )
        else:
             raise ValueError(f"Invalid encoder type: {self.config.encoder_type}")

        # Freeze the backbone
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.image_encoder.eval()

        self.feature_key = 'x_norm_patchtokens'
        self.emb_dim = self.image_encoder.num_features
        if self.feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif self.feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {self.feature_key}")

        self.patch_size = self.image_encoder.patch_size

    def forward(self, image):
        """Encode normalized images without resizing or cropping their pixels."""
        # 256x512 needs no padding with DINOv3: 16x32 patches, matching JEPA-WM.
        height, width = image.shape[-2:]
        image_features = F.pad(image, (0, -width % self.patch_size, 0, -height % self.patch_size))
        emb = self.image_encoder.forward_features(image_features)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim

        return emb

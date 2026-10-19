import logging
import os
from pathlib import Path

import torch
from torch import nn

from navsim.agents.le_wm.le_wm_config import LE_WMConfig


logger = logging.getLogger(__name__)


class LE_WMBackbone(nn.Module):
    """Native DINOv3 initialized from scratch, returning one CLS token per frame."""

    def __init__(self, config: LE_WMConfig):
        super().__init__()
        self.config = config

        model_kwargs = self.config.model
        encoder_names = {
            "vit-s": "dinov3_vits16",
            "vit-b": "dinov3_vitb16",
            "vit-l": "dinov3_vitl16",
        }
        if model_kwargs["encoder_scale"] not in encoder_names:
            raise ValueError(f"Invalid encoder type: {model_kwargs['encoder_scale']}")
        if model_kwargs["patch_size"] != 16:
            raise ValueError("The native DINOv3 backbones require patch_size=16.")
        encoder_name = encoder_names[model_kwargs["encoder_scale"]]

        # Use the same native hub constructor as JEPA-WM and the downstream
        # agent, including its RoPE, register tokens, normalization and LayerScale.
        project_root = Path(__file__).resolve().parents[3]
        repo_candidates = []
        if os.environ.get("JEPAWM_HOME"):
            repo_candidates.append(Path(os.environ["JEPAWM_HOME"]) / "dinov3")
        repo_candidates.extend([
            Path(os.environ.get("NAVSIM_DEVKIT_ROOT", str(project_root))) / "dinov3",
            project_root / "download" / "deps" / "dinov3",
            Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main",
        ])
        repo_path = next(
            (path for path in repo_candidates if (path / "hubconf.py").is_file()), None
        )
        if repo_path is not None:
            self.base_model = torch.hub.load(
                str(repo_path), encoder_name, source="local", pretrained=False
            )
        else:
            self.base_model = torch.hub.load(
                "facebookresearch/dinov3", encoder_name, pretrained=False, trust_repo=True
            )

        # pretrained=False invokes DINOv3's native init_weights(), without loading
        # pretrained weights. Keep base_model naming compatible with DinoEncoder.
        self.base_model.requires_grad_(True)
        self.emb_dim = self.base_model.num_features
        logger.info("Initialized LE-WM encoder %s with random weights (pretrained=False).", encoder_name)

    def forward(self, image):
        """
        Forward pass to extract [CLS] token embedding.
        Args:
            image (torch.Tensor): (B, 3, H, W)
        Returns:
            emb (torch.Tensor): (B, D)
        """
        # Preserve LE-WM's CLS objective; the same base_model also exposes
        # x_norm_patchtokens for downstream perception-free trajectory training.
        return self.base_model.forward_features(image.float())["x_norm_clstoken"]

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from einops import rearrange, repeat

from navsim.agents.jepa_wm_v2.jepa_wm_config import JEPA_WMConfig
from navsim.agents.jepa_wm_v2.jepa_wm_features import BoundingBox2DIndex
from navsim.agents.jepa_wm_v2.jepa_wm_projector import (
    build_visual_projector,
    get_visual_projector_output_dim,
)
from navsim.agents.lejepa_sigreg import SIGReg
from navsim.common.enums import StateSE2Index


import sys
import yaml
import os
from pathlib import Path

# Add jepa-wms to sys.path so we can import app modules
CURRENT_DIR = Path(__file__).resolve().parent
JEPA_WM_DIR = CURRENT_DIR / "jepa-wms"
if str(JEPA_WM_DIR) not in sys.path:
    sys.path.insert(0, str(JEPA_WM_DIR))

from app.vjepa_wm.utils import init_video_model
from app.vjepa_wm.video_wm import VideoWM


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


class JEPA_WMModel(nn.Module):
    """Torch module for JEPA_WM wrapping jepa-wms VideoWM."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: JEPA_WMConfig):
        """
        Initializes JEPA_WM torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of JEPA_WM.
        """
        super().__init__()
        self._config = config
            
        model_kwargs = self._config.model
        self.detach_targets = _as_bool(model_kwargs.get("detach_targets", True))
        self.sigreg_patch_weight = float(model_kwargs.get("sigreg_patch_weight", 0.0))
        self._projected_grid_size = tuple(model_kwargs["projected_grid_size"])
        
        self.num_hist = 4 # past 4 frames
        self.num_pred = model_kwargs.get("num_frames_pred", 1) # default to predicting whatever is in config

        # Extract features necessary for our wrappers
        self.action_dim = model_kwargs.get("action_encoder", {}).get("action_emb_dim", 0) * (model_kwargs.get("action_encoder", {}).get("action_tokens", 0) if model_kwargs.get("action_conditioning", "") == "token" else 1)
        self.proprio_dim = model_kwargs.get("proprio_encoder", {}).get("proprio_emb_dim", 0) * (model_kwargs.get("proprio_encoder", {}).get("proprio_tokens", 0) if model_kwargs.get("proprio_encoding", "") == "token" else 1)

        
        # Some defaults that were used in the old model logic for concatenation
        self.concat_dim = 1
        self.num_action_repeat = 1
        self.num_proprio_repeat = 1
        visual_encoder_cfg = model_kwargs.get("visual_encoder", {})
        self.visual_projector = build_visual_projector(visual_encoder_cfg)
        self.visual_projector_output_dim = get_visual_projector_output_dim(visual_encoder_cfg)
        # Determine device appropriately, though typically models will be moved to cuda later
        device = "cpu"
        
        # Extract init_video_model kwargs by flattening nested configs
        from omegaconf import DictConfig, OmegaConf
        
        init_model_kwargs = {}
        for k, v in model_kwargs.items():
            if isinstance(v, DictConfig):
                v = OmegaConf.to_container(v, resolve=True)
            if k == "attn":
                init_model_kwargs["cfgs_attn_pattern"] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    init_model_kwargs[k2] = v2
            else:
                init_model_kwargs[k] = v
        # Ensure action and proprio dimensions are set (defaults based on navsim specifications)
        if "action_dim" not in init_model_kwargs:
            init_model_kwargs["action_dim"] = 3
        if "proprio_dim" not in init_model_kwargs:
            init_model_kwargs["proprio_dim"] = 8
        if "img_size" not in init_model_kwargs:
            # DINOv3 ViT-L16 has patch size 16. grid_size 14 implies img_size = 14 * 16 = 224
            init_model_kwargs["img_size"] = model_kwargs.get("grid_size", 14) * 16
        init_model_kwargs["embed_dim"] = self.visual_projector_output_dim
        init_model_kwargs["pred_embed_dim"] = self.visual_projector_output_dim
            
        predictor_init_kwargs = dict(init_model_kwargs)
        # DINO keeps the raw image grid; the patch predictor operates on the projected grid.
        predictor_init_kwargs["img_size"] = [
            image_size * projected_size // grid_size
            for image_size, grid_size, projected_size in zip(
                model_kwargs["img_size"],
                model_kwargs["grid_size"],
                self._projected_grid_size,
            )
        ]
        predictor, encoder, action_encoder, proprio_encoder = init_video_model(
            device=device, **predictor_init_kwargs
        )
        encoder.requires_grad_(False)
        
        # Filter init_model_kwargs to only include arguments valid for VideoWM
        import inspect
        sig = inspect.signature(VideoWM.__init__)
        valid_wm_kwargs = {k: v for k, v in init_model_kwargs.items() if k in sig.parameters}

        # Initialize VideoWM
        self.video_wm = VideoWM(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            **valid_wm_kwargs
        )
        self.emb_criterion = nn.MSELoss() 
        self.sigreg = SIGReg(
            knots=int(model_kwargs.get("sigreg_knots", 17)),
            num_proj=int(model_kwargs.get("sigreg_num_proj", 1024)),
        )
    
    def encode(self, obs, act):
        """
        input :  obs (dict): "visual": (b, num_frames, 3, height, width) , "proprio": (b, num_frames, 8);
                 act: (b, num_frames, 3)
        output: patch embeddings (b, t, num_patches, dim).
        """
        # VideoWM's encode requires visual to be shape (B, T, V=1, C=3, H, W) mostly, but we use the existing dict
        # Actually in encode_obs: (b, t, c, h, w)
        video_features, proprio_features, action_features = self.video_wm.encode(obs, act)

        b, t, v, h, w, d = video_features.shape
        video_features_2d = rearrange(video_features, "b t v h w d -> (b t v) d h w")
        video_features_2d = self.visual_projector(video_features_2d)
        if video_features_2d.shape[1] != self.visual_projector_output_dim:
            raise ValueError(
                f"Projector output dim {video_features_2d.shape[1]} must match configured projected dim "
                f"{self.visual_projector_output_dim}."
            )
        video_features = rearrange(video_features_2d, "(b t v) d h w -> b t v h w d", b=b, t=t, v=v)
        
        # Flatten the projected front-camera grid into patch tokens.
        b, t, v, h, w, d = video_features.shape
        if v != 1:
            raise ValueError(f"JEPA-WM expects one front-camera view, got {v} views")
        # video_features is (b, t, 1, h, w, d) => strip view and flatten spatial
        z_vis = video_features.reshape(b, t, h*w, d)
        
        # For proprio and act, they come out as (b, t, 1, a_d) or (b, t, h*w, a_d) depending on conditioning
        # Let's concatenate them according to how jepa_wm_agent uses it
        
        if proprio_features is not None:
            if proprio_features.dim() == 3:
                proprio_features = proprio_features.unsqueeze(2)
            if proprio_features.shape[2] == 1:
                from einops import repeat
                proprio_features = repeat(proprio_features, "b t 1 d -> b t p d", p=h*w)
            
        if action_features is not None:
            if action_features.dim() == 3:
                action_features = action_features.unsqueeze(2)
            if action_features.shape[2] == 1:
                from einops import repeat
                action_features = repeat(action_features, "b t 1 d -> b t p d", p=h*w)
        
        if self.video_wm.pred_type == "AdaLN":
            # AdaLN does not expect concatenation here, it expects visual features only
            z = z_vis
        else:
            # Build the final z (concatenated)
            z = self.video_wm.concat_obs_act(z_vis, action_features, proprio_features)
        
        return z

    def predict(self, z, action_features=None, proprio_features=None):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        
        if self.video_wm.pred_type == "AdaLN":
            # AdaLN expects z to be (B, T, V, H, W, D) or we can spoof V, H, W
            # Actually AdaLN predictor expects (B, T, V, H, W, D) input shape based on forward_pred
            # Reconstruct dummy V,H,W dimensions. V=1 usually.
            H, W = self._projected_grid_size
            z = rearrange(z, "b t (h w) d -> b t 1 h w d", h=H, w=W)
            
            # Predict
            pred_video_features, _, _ = self.video_wm.predictor(
                z, action_features, proprio_features
            )
            
            # Predictor already outputs (B, T, H*W, D)
            z = pred_video_features
            
        elif self.video_wm.pred_type == "dino_wm":
            z = rearrange(z, "b t p d -> b (t p) d")
            z = self.video_wm.predictor(z)
            z = rearrange(z, "b (t p) d -> b t p d", t=T)
            
        else:
            raise NotImplementedError(f"Prediction for {self.video_wm.pred_type} not handled in JEPA_WMModel.predict")
            
        return z

    def advance_rollout_context(
        self,
        context: torch.Tensor,
        prediction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the newest prediction to a fixed-length rollout context."""
        next_prediction = prediction[:, -1:]
        next_context = torch.cat([context[:, 1:], next_prediction], dim=1)
        return next_context, next_prediction

    def encode_act(self, act):
        # Match expected interface from agent
        return self.video_wm.encode_act(act)

    def encode_proprio(self, proprio):
        # Match expected interface from agent
        return self.video_wm.encode_proprio(proprio)

    def visual_only(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        return camera_feature

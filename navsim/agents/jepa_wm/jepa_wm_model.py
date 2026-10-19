import inspect
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import DictConfig, OmegaConf

from navsim.agents.jepa_wm.jepa_wm_config import JEPA_WMConfig

# Keep the upstream implementation unchanged; adapt its interface here.
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
        self.unfreeze_encoder = _as_bool(model_kwargs.get("unfreeze_encoder", False))
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
        self.visual_embed_dim = int(visual_encoder_cfg["embed_dim"])

        # Determine device appropriately, though typically models will be moved to cuda later
        device = "cpu"
        
        # Extract init_video_model kwargs by flattening nested configs
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
        # Predictor input/output dimensions and patch grid remain in DINO space.
        predictor, encoder, action_encoder, proprio_encoder = init_video_model(
            device=device, **init_model_kwargs
        )
        if self.unfreeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = True
        
        # Filter init_model_kwargs to only include arguments valid for VideoWM
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

    def encode(self, obs, act):
        """Encode frames as original DINO patch embeddings, without projection."""
        video_features, proprio_features, action_features = self.video_wm.encode(obs, act)
        z_vis = rearrange(video_features, "b t 1 h w d -> b t (h w) d")
        if self.video_wm.pred_type == "AdaLN":
            return z_vis
        h, w = self.video_wm.grid_height, self.video_wm.grid_width

        # For proprio and act, they come out as (b, t, 1, a_d) or (b, t, h*w, a_d) depending on conditioning
        # Let's concatenate them according to how jepa_wm_agent uses it
        
        if proprio_features is not None:
            if proprio_features.dim() == 3:
                proprio_features = proprio_features.unsqueeze(2)
            if proprio_features.shape[2] == 1:
                proprio_features = repeat(proprio_features, "b t 1 d -> b t p d", p=h*w)
            
        if action_features is not None:
            if action_features.dim() == 3:
                action_features = action_features.unsqueeze(2)
            if action_features.shape[2] == 1:
                action_features = repeat(action_features, "b t 1 d -> b t p d", p=h*w)
        
        return self.video_wm.concat_obs_act(z_vis, action_features, proprio_features)

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
            H, W = self.video_wm.grid_height, self.video_wm.grid_width
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

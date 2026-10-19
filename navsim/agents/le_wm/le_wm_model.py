from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.le_wm.le_wm_config import LE_WMConfig
from navsim.agents.le_wm.le_wm_backbone import LE_WMBackbone
from navsim.agents.le_wm.le_wm_module import Embedder, ARPredictor, MLP
from navsim.agents.lejepa_sigreg import SIGReg


class LE_WM_Model(nn.Module):
    """Torch module for LE_WM."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: LE_WMConfig):
        """
        Initializes LE_WM torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of LE_WM.
        """
        super().__init__()

        self._config = config

        model_kwargs = self._config.model
        self.num_hist = model_kwargs["history_size"]
        self.num_pred = model_kwargs["num_preds"]

        self.backbone = LE_WMBackbone(config)
        
        self.embed_dim = model_kwargs["embed_dim"]
        
        # Project the ViT CLS token to the regularized latent dimension.
        self.projector = MLP(
            input_dim=self.backbone.emb_dim,
            output_dim=self.embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d
        )
        
        self.pred_proj = MLP(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d
        )

        # Encode relative SE(2) actions with LeWorld's Conv1d + MLP.
        self.action_encoder = Embedder(input_dim=3, emb_dim=self.embed_dim)

        self.predictor = ARPredictor(
            num_frames=self.num_hist,
            depth=model_kwargs["predictor_depth"],
            heads=model_kwargs["predictor_heads"],
            mlp_dim=model_kwargs["predictor_mlp_dim"],
            input_dim=self.embed_dim,
            hidden_dim=self.embed_dim,
            output_dim=self.embed_dim,
            dim_head=model_kwargs["predictor_dim_head"],
            dropout=model_kwargs["predictor_dropout"],
            emb_dropout=model_kwargs["predictor_emb_dropout"],
        )

        # LEWM SIGReg loss
        self.sigreg = SIGReg(knots=model_kwargs["sigreg_knots"], num_proj=model_kwargs["sigreg_num_proj"])
        self.sigreg_weight = model_kwargs["sigreg_weight"]
    
    def encode_act(self, act):
        """
        act: (B, T, 3)
        Returns action conditioning (B, T, predictor hidden dim)
        """
        return self.action_encoder(act)

    def encode_obs(self, obs):
        """
        obs (dict): "visual" (b, t, 3, h, w)
        Returns z (tensor) : (B, T, D) sequence embeddings
        """
        visual = obs['visual']
        b, t = visual.shape[:2]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        
        # Get CLS token embeddings (BS, D_encoder)
        emb = self.backbone(visual)
        
        # Project to target dimension
        emb = self.projector(emb)
        
        # Reshape back to sequence (B, T, D)
        emb = rearrange(emb, "(b t) d -> b t d", b=b, t=t)
        return emb

    def predict(self, emb, act_emb):
        """
        Predict next latent states.
        emb: (B, T, D)
        act_emb: (B, T, predictor hidden dim)
        Returns preds (B, T, D)
        """
        preds = self.predictor(emb, act_emb)
        
        # Map back to embedding space
        b = preds.size(0)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=b)
        
        return preds

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        camera_feature: torch.Tensor = features["camera_feature"]
        return camera_feature

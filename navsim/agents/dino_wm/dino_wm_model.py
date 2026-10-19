from typing import Dict

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.dino_wm.dino_wm_backbone import DINO_WMBackbone
from navsim.agents.dino_wm.dino_wm_action_encoder import ProprioceptiveEmbedding
from navsim.agents.dino_wm.dino_wm_predictor import ViTPredictor

from navsim.agents.dino_wm.dino_wm_config import DINO_WMConfig


from einops import rearrange, repeat

class DINO_WMModel(nn.Module):
    """Torch module for DINO_WM."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: DINO_WMConfig):
        """
        Initializes DINO_WM torch module.
        :param trajectory_sampling: trajectory sampling specification.
        :param config: global config dataclass of DINO_WM.
        """

        super().__init__()

        self._config = config
        self.num_hist = 4 # use past 4 frames as history
        self.num_pred = 1 # only predict next frame
        self.backbone = DINO_WMBackbone(config)
        self.action_encoder = ProprioceptiveEmbedding(num_frames=4, in_chans=3, emb_dim=config.action_embed_dim)
        self.num_action_repeat = 1
        action_dim = config.action_embed_dim
        self.action_dim = action_dim * self.num_action_repeat 
        self.embedding_dim = self.backbone.emb_dim + self.action_dim
        self.grid_size = tuple(
            (size + self.backbone.patch_size - 1) // self.backbone.patch_size
            for size in config.img_size
        )
        num_patches = self.grid_size[0] * self.grid_size[1]
        depth = config.predictor_depth

        if self._config.predictor_size == 'small':
            self.predictor = ViTPredictor(num_patches=num_patches, num_frames=4, encoder_dim=self.embedding_dim, dim=384, depth=depth, heads=6, mlp_dim=384*4)
        elif self._config.predictor_size == 'base':
            self.predictor = ViTPredictor(num_patches=num_patches, num_frames=4, encoder_dim=self.embedding_dim, dim=768, depth=depth, heads=12, mlp_dim=768*4)
        elif self._config.predictor_size == 'large':
            self.predictor = ViTPredictor(num_patches=num_patches, num_frames=4, encoder_dim=self.embedding_dim, dim=1024, depth=depth, heads=16, mlp_dim=1024*4)
        elif self._config.predictor_size == 'huge':
            self.predictor = ViTPredictor(num_patches=num_patches, num_frames=4, encoder_dim=self.embedding_dim, dim=1280, depth=depth, heads=16, mlp_dim=1280*4)
        elif self._config.predictor_size == 'giant':
            self.predictor = ViTPredictor(num_patches=num_patches, num_frames=4, encoder_dim=self.embedding_dim, dim=1536,  depth=depth, heads=24, mlp_dim=1536*4)
        else:
            raise ValueError(f"Invalid predictor size: {self._config.predictor_size}")
        

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"encoder: {self.backbone}")
        print(f"action encoder: {self.action_encoder}")
        print(f"predictor: {self.predictor}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        self.concat_dim = 1 # channel fusion: concatenate the action embedding to each patch's embedding

        self.emb_criterion = nn.MSELoss() 
    
    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual": (b, num_frames, 3, 256, 512);
                 act: (b, num_frames, 3), one outgoing SE(2) transition per frame
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        return self.concat_obs_act(z_dct['visual'], act_emb)

    def concat_obs_act(self, visual_embs, act_emb):
        """Attach the current transition action to every patch before prediction."""
        if self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual_embs.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [visual_embs, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        else:
            raise Exception("concat_dim 0 not supported yet.")
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual" (b, t, 3, h, w)
        output:   z (dict): "visual",(b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual_embs = self.backbone(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        return {"visual": visual_embs}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def advance_rollout_context(self, context, prediction):
        """Retain three visual context frames and append the newest prediction."""
        next_prediction = prediction[:, -1:, :, :-self.action_dim]
        next_context = torch.cat([context[:, 1:], next_prediction], dim=1)
        return next_context, next_prediction
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 1:
            z_visual, z_act = z[..., :-self.action_dim], z[..., -self.action_dim:]
            # remove tiled dimensions
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        else:
            raise Exception("concat_dim 0 not supported yet.")
        z_obs = {"visual": z_visual}
        return z_obs, z_act


    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        # just forward the camerea feature

        camera_feature: torch.Tensor = features["camera_feature"]

        return camera_feature
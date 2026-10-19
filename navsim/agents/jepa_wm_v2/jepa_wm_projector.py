from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ChannelProjector(nn.Module):
    """Apply lightweight 2D convolutions on per-frame spatial encoder features."""

    def __init__(self, in_dim: int, conv_layers: Optional[list[Any]] = None, norm_type: str = "layer"):
        super().__init__()
        self.norm_type = norm_type
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for cfg in conv_layers or []:
            out_ch = int(_cfg_get(cfg, "out_dim"))
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=int(_cfg_get(cfg, "in_dim")),
                    out_channels=out_ch,
                    kernel_size=int(_cfg_get(cfg, "kernel_size")),
                    stride=int(_cfg_get(cfg, "stride")),
                    padding=int(_cfg_get(cfg, "padding")),
                )
            )
            if self.norm_type == "batch":
                self.norm_layers.append(nn.BatchNorm2d(out_ch))
            elif self.norm_type == "layer":
                self.norm_layers.append(nn.LayerNorm(out_ch))
            else:
                self.norm_layers.append(nn.Identity())

        self.out_dim = int(_cfg_get((conv_layers or [])[-1], "out_dim", in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C_out, H, W)
        """
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            norm = self.norm_layers[i]
            if self.norm_type == "batch":
                x = norm(x)
            elif self.norm_type == "layer":
                x = x.permute(0, 2, 3, 1)
                x = norm(x)
                x = x.permute(0, 3, 1, 2)
            else:
                x = norm(x)
            if i != len(self.conv_layers) - 1:
                x = F.gelu(x)
        return x


def build_visual_projector(visual_encoder_cfg: Any) -> nn.Module:
    """Build the channelwise patch projector."""
    projector_cfg = _cfg_get(visual_encoder_cfg, "projector_config")
    if projector_cfg is None:
        raise ValueError("visual_encoder.projector_config is required")

    in_dim = int(_cfg_get(visual_encoder_cfg, "embed_dim"))
    return ChannelProjector(
        in_dim=in_dim,
        conv_layers=_cfg_get(projector_cfg, "conv_layers"),
        norm_type=_cfg_get(projector_cfg, "norm_type", "layer"),
    )


def get_visual_projector_output_dim(visual_encoder_cfg: Any) -> int:
    projector_cfg = _cfg_get(visual_encoder_cfg, "projector_config")
    conv_layers = _cfg_get(projector_cfg, "conv_layers", [])
    if conv_layers:
        return int(_cfg_get(conv_layers[-1], "out_dim"))
    return int(_cfg_get(visual_encoder_cfg, "embed_dim"))

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class DINO_WMConfig:
    """Global DINO_WM config."""
    encoder_type: str = 'vit-l'
    predictor_size: str = 'large'
    predictor_depth: int = 12
    img_size: Tuple[int, int] = (256, 512)
    action_embed_dim: int = 10
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        "transition_model": {
            "weight_decay": 1e-7,
            "betas": (0.9, 0.995),
            "eps": 1e-8,
        }
    })

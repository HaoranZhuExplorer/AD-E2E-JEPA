from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ViTPerceptionFreeConfig:
    """Configuration shared by the ViT perception-free agent components."""

    model: Dict[str, Any] = field(default_factory=dict)

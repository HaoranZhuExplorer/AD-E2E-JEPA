from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer

@dataclass
class JEPA_WMConfig:
    """Global JEPA_WM config."""

    # Model configuration extracted from hydra yaml
    model: Dict[str, Any] = field(default_factory=dict)

    # Optimization config extracted from hydra yaml
    optimization: Dict[str, Any] = field(default_factory=dict)

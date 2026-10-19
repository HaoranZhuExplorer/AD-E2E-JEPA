from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import ImageColor

from navsim.agents.jepa_wm.jepa_wm_config import JEPA_WMConfig
from navsim.agents.jepa_wm.jepa_wm_features import BoundingBox2DIndex
from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG


class JEPA_WMCallback(pl.Callback):
    """Visualization Callback for JEPA_WM during training."""

    def __init__(
        self,
        config: JEPA_WMConfig,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        """
        Initializes the visualization callback.
        :param config: global config dataclass of JEPA_WM
        :param num_plots: number of images tiles, defaults to 3
        :param num_rows: number of rows in image tile, defaults to 2
        :param num_columns: number of columns in image tile, defaults to 2
        """

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        
    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""
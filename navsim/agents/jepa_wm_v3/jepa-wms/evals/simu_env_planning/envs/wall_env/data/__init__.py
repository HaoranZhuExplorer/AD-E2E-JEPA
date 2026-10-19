# Copyright (c) Facebook, Inc. and its affiliates.
# Inspired from https://github.com/gaoyuezhou/dino_wm
# Licensed under the MIT License

from enum import Enum, auto


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    WallEigenfunc = auto()

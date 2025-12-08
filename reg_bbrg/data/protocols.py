import torch
from enum import Enum
from monai.transforms.spatial.array import Resize
from monai.transforms.croppad.array import SpatialPad
from typing import TypedDict


class ImageDimension(Enum):
    D2 = 2
    """2D images."""
    D3 = 3
    """3D images."""


class MonaiData(TypedDict):
    image: torch.Tensor


class MedicalTranslationData(TypedDict):
    input: torch.Tensor
    target: torch.Tensor


class ResizeMode(Enum):
    RESIZE = Resize
    """Resize the image to the specified size."""
    PAD = SpatialPad
    """Pad the image with zeros to the specified size."""

from enum import Enum
from typing import NamedTuple


class UNetConfigs(NamedTuple):
    attention_resolutions: tuple[int, ...]
    dim_mults: tuple[int, ...]


class UNetType(Enum):
    """The type of UNet."""
    BBDM = UNetConfigs((32, 16, 8), (1, 4, 8))
    A_BRIDGE = UNetConfigs((32, 16, 8), (1, 2, 4, 8))
    FDDPM = UNetConfigs((16,), (1, 1, 2, 2, 4, 4))

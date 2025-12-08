from copy import deepcopy
from torchmanager.data import Dataset
from torchmanager_core import abc, torch
from torchmanager_core.typing import Any


class VolumeToSliceDataset(Dataset[torch.Tensor], abc.ABC):
    """
    Base class for slicing 3D volume into 2D slices.

    * extends: `torchmanager.data.Dataset`.
    * abstract class
    """
    @abc.abstractmethod
    def __getitem__(self, index: int) -> Any:
        """
        Load the data at the given index. 3D volumes need to be returned in format of `S x C x H x W`.

        * Abstract method to implement.

        - Parameters:
            - index: The index of the data in `int`.
        - Returns: A `dict` of the input and target images in `torch.Tensor`.
        """
        ...

    @staticmethod
    def unpack_data(volumes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # combine batch dimension and depth dimension
        slices = volumes.view(-1, *volumes.shape[2:])  # [B, N, 1, ...] -> [B * N, 1, ...]
        return deepcopy(volumes), deepcopy(slices)

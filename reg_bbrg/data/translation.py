import math
from monai.data.dataset import Dataset as MonaiDataset
from monai.transforms.spatial.array import Resize
from monai.transforms.croppad.array import SpatialPad
from torchmanager_core import devices, torch
from torchmanager_core.typing import cast

from .protocols import ImageDimension, MonaiData, ResizeMode, MedicalTranslationData
from .volume import VolumeToSliceDataset


class MedicalTranslationDataset(VolumeToSliceDataset):
    """
    A Paired dataset for image translation

    * extends: `.base.MedicalDataset`

    - Parameters:
        - dimension: The image dimension in `ImageDimension`
        - image_size: The image size in `tuple[int, int, int]`
        - input_modality: The input modality to use in `int`
        - monai_dataset: A `MonaiDataset` that contains the data
        - resize: The resize transform to use in `Resize` or `SpatialPad`
        - target_modality: The target modality to use in `int`
        - use_slices: A `bool` flag to count slices as mini-batches
    """
    dimension: ImageDimension
    image_size: tuple[int, int, int]
    input_modality: int | list[int]
    monai_dataset: MonaiDataset
    resize: Resize | SpatialPad
    target_modality: int
    use_slices: bool

    def __init__(self, d: MonaiDataset, batch_size: int, image_size: tuple[int, int, int], /, input_modality: int | list[int] = 0, target_modality: int = 1, dimension: ImageDimension = ImageDimension.D2, resize_mode: ResizeMode = ResizeMode.PAD, *, device: torch.device = devices.CPU, drop_last: bool = False, num_workers: int | None = None, shuffle: bool = False, use_slices: bool = False) -> None:
        """
        Initialize the translation dataset.

        - Parameters:
            - d: The `MonaiDataset` that contains the data.
            - batch_size: The batch size in `int`.
            - image_size: The image size in `tuple[int, int, int]`.
            - input_modality: The input modality index to use in `int` or a `list` of `int`.
            - target_modality: The target modality index to use in `int`.
            - dimension: The image dimension in `ImageDimension`.
            - resize_mode: The resize mode to use in `ResizeMode`.
            - device: The device to load the data on in `torch.device`.
            - drop_last: The flag to drop the last batch in `bool`.
            - num_workers: The number of workers to use in `int`.
            - shuffle: The flag to shuffle the data in `bool`.
            - use_slices: The flag to count slices as mini-batches in `bool``.
        """
        super().__init__(batch_size, device=device, drop_last=drop_last, num_workers=num_workers, shuffle=shuffle)
        self.dimension = dimension
        self.image_size = image_size
        self.input_modality = input_modality
        self.monai_dataset = d
        self.target_modality = target_modality
        self.use_slices = use_slices

        # initialize resize mode
        self.resize = resize_mode.value(image_size)

        # check slice size not to be give in 3D
        if dimension == ImageDimension.D3:
            assert not use_slices, "Slice is not supported in 3D"

    @property
    def unbatched_len(self) -> int:
        # slice size is 1 if not given otherwise separate the slice size of image size into mini-batches, do not skip the last batch
        slice_size = self.image_size[2] if self.use_slices else 1
        # ceiling slice size
        slice_size = math.ceil(slice_size)
        return len(self.monai_dataset) * slice_size

    @torch.no_grad()
    def __getitem__(self, index: int) -> list[MedicalTranslationData]:
        # calculate index if slice size is given
        if self.use_slices:
            # calculate the index of the image
            image_index = index // self.image_size[2]
            # calculate the index of the slice
            slice_index = index % self.image_size[2]

            # get data from monai dataset
            data = self.monai_dataset[image_index]  # type: ignore
        else:
            # get data from monai dataset
            data = self.monai_dataset[index]  # type: ignore
            slice_index = 0

        # cast data type
        data = cast(MonaiData | list[MonaiData], data)

        # check data type
        if isinstance(data, dict):
            data_list: list[MonaiData] = [data]
        elif isinstance(data, list):
            data_list = data
            assert isinstance(data_list, list)

        # get the image and target modalities
        translation_data_list: list[MedicalTranslationData] = []
        for d in data_list:
            # sliding window for input and target
            d['image'] = self.resize(d['image'])

            # get slices from slice index to slice index + slice size, [4, H, W, D] -> [4, H, W, 1]
            if self.use_slices:
                d['image'] = d['image'][..., slice_index:slice_index + 1]

            # check dimension
            match self.dimension:
                case ImageDimension.D2:
                    # move depth dimension to first
                    d['image'] = d['image'].permute(3, 0, 1, 2)  # [4, H, W, D] -> [D, 4, H, W]
                case ImageDimension.D3:
                    d['image'] = d['image'].unsqueeze(0)  # [4, H, W, D] -> [1, 4, H, W, D]

            # normalize image to [0, 1]
            d['image'] = (d['image'] - d['image'].min()) / (d['image'].max() - d['image'].min())
            d['image'] = torch.nan_to_num(d['image'], nan=0)
            d['image'] = d['image'] * 2 - 1  # normalize to [-1, 1]

            # extract the input and target modalities
            target_image = d['image'][:, self.target_modality, ...].unsqueeze(1)  # [N, 4, ...] -> [N, 1, ...]
            input_image = d['image'][:, self.input_modality, ...]  # [N, 4, ...] -> [N, M, ...] or [N, ...]
            if not isinstance(self.input_modality, list):
                input_image = input_image.unsqueeze(1)  # [N, ...] -> [N, 1, ...]

            # append the data
            translation_data = MedicalTranslationData(input=input_image, target=target_image)
            translation_data_list.append(translation_data)
        return translation_data_list

    @staticmethod
    def unpack_data(volumes_dict: MedicalTranslationData) -> tuple[torch.Tensor, torch.Tensor]:
        input_volume, target_volume = volumes_dict['input'], volumes_dict['target']
        _, input_slice = VolumeToSliceDataset.unpack_data(input_volume)
        _, target_slice = VolumeToSliceDataset.unpack_data(target_volume)
        return input_slice, target_slice

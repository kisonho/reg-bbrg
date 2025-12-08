# pyright: reportPrivateImportUsage=false
import imageio, numpy as np, os, torch
from enum import Enum
from monai.data.dataset import Dataset
from monai import transforms
from monai.transforms.compose import Compose
from numpy.typing import NDArray
from typing import Any, NamedTuple, Optional, Union


class ImageType(Enum):
    HDR = ".hdr"
    IMG = ".img"


class ISeg(Dataset):
    '''
    The iSeg dataset

    - Properties:
        - data: A `list` of the prefix for all data in the given dataset without modality, i.e. \'subject-1\'
        - modality: The target `ImageModality`
    '''
    root_dir: str
    image_type: ImageType
    '''Target modality, the modality will be set to the first channel in the image'''

    def __init__(self, root_dir: str, data_prefix: list[str] = [f"subject-{i}" for i in range(1, 11)], shuffle: bool = False, transform: Optional[Compose] = None, type: ImageType = ImageType.IMG) -> None:
        '''
        Constructor

        - Parameters:
            - root_dir: A `str` of the root directory
            - data_prefix: A `list` of the prefix for all data in the given dataset without modality, i.e. \'subject-1\'
            - shuffle: A `bool` flag of if shuffling the dataset
            - transorms: An optional `Compose` to transform the input image
            - type: An `ImageType` of the data extension
        '''
        super().__init__(data_prefix, transform=transform)
        self.root_dir = os.path.normpath(root_dir)
        self.image_type = type
        if shuffle:
            np.random.shuffle(self.data)

    @torch.no_grad()
    def __getitem__(self, i: int) -> Union[list[dict[str, Any]], dict[str, Any]]:
        # read label
        label_file = f"{self.data[i]}-label{self.image_type.value}"
        label_path = os.path.join(self.root_dir, label_file)
        label: NDArray[np.int_] = imageio.imread(label_path)

        # reat t1 and t2 images
        t1_file = f"{self.data[i]}-T1{self.image_type.value}"
        t1_file_path = os.path.join(self.root_dir, t1_file)
        with imageio.get_reader(t1_file_path) as reader:
            t1_img = reader.get_data(0)  # type: ignore
        # t1_img = imageio.imread(t1_file_path)
        t2_file = f"{self.data[i]}-T2{self.image_type.value}"
        t2_file_path = os.path.join(self.root_dir, t2_file)
        with imageio.get_reader(t2_file_path) as reader:
            t2_img = reader.get_data(0)  # type: ignore
        image = [t1_img, t2_img]

        # wrap data
        data = {
            'image': np.array(image),
            'label': np.array([label]),
        }

        # transform image
        if self.transform is not None:
            data: list[dict[str, Any]] | dict[str, Any] = self.transform(data) # type: ignore
        return data

    def __len__(self) -> int:
        return len(self.data)

    def split(self, p: float | int) -> tuple['ISeg', 'ISeg']:
        '''
        Split current dataset into two `ISeg` dataset
        
        - Parameters:
            - p: A `float` of split ratio
        - Returns A `tuple` of two `ISeg` dataset with the first one contains p ratio of current and another contains 1-p ratio of current
        '''
        # initialize
        if p > 0 and p < 1:
            split_amount = int(len(self) * p)
        elif p >= 1:
            split_amount = int(p)
        else:
            raise ValueError("Split ratio must be greater than 0.")

        # split data
        first_dataset = ISeg(self.root_dir, data_prefix=[d for d in self.data[:split_amount]], transform=self.transform, type=self.image_type)
        second_dataset = ISeg(self.root_dir, data_prefix=[d for d in self.data[split_amount:]], transform=self.transform, type=self.image_type)
        return first_dataset, second_dataset


class TransformOptions(NamedTuple):
    load_imaged: bool = True
    """must be `True`"""
    center_spatial_cropd: Optional[Union[tuple[int, int, int], int]] = None
    spacingd: Optional[tuple[tuple[int, ...], str]] = None
    """`tuple`[pixdim (`tuple`), mode (`int`)]"""
    normalize_intensityd: bool = False
    crop_foregroundd: bool = False
    grid_patchd: Optional[tuple[int, ...]] = None
    spatial_padd: tuple[int, ...] | None = None
    orientationd: bool = False
    rand_crop_by_pos_neg_labeld: Optional[tuple[int, ...]] = None
    rand_flipd: bool = False
    rand_rotate_90d: bool = False
    rand_scale_intensityd: bool = False
    rand_shift_intensityd: bool = False
    to_tensord: bool = False


def load_transforms(transform_options: TransformOptions, keys: list[str]) -> tuple[Compose, Compose]:
    """
    Load transforms via options

    - Parameters:
        - transform_options: A `TransformOptions` of transform settings
        - img_size: A `tuple` of target image size in `int`
        - keys: A `list` of input keys in `str`
        - mapping_dict: A `dict` of label value mapping
    """
    # load transforms
    training_transforms: list[transforms.Transform] = []

    if transform_options.load_imaged:
        training_transforms.append(transforms.LoadImaged(keys))

    if transform_options.center_spatial_cropd is not None:
        training_transforms.append(transforms.CenterSpatialCropd(keys, transform_options.center_spatial_cropd))

    if transform_options.spacingd is not None:
        training_transforms.append(transforms.Spacingd(keys=keys, pixdim=transform_options.spacingd[0], mode=transform_options.spacingd[1]))

    if transform_options.orientationd:
        training_transforms.append(transforms.Orientationd(keys=keys, axcodes="RAS"))

    if transform_options.normalize_intensityd:
        training_transforms.append(transforms.NormalizeIntensityd(keys=keys[0], channel_wise=True))

    if transform_options.crop_foregroundd:
        training_transforms.append(transforms.CropForegroundd(keys=keys, source_key=keys[0]))

    if transform_options.spatial_padd is not None:
        training_transforms.append(transforms.SpatialPadd(keys=keys, spatial_size=transform_options.spatial_padd))

    validation_transforms = training_transforms.copy()

    if transform_options.rand_crop_by_pos_neg_labeld is not None:
        training_transforms.append(transforms.RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=transform_options.rand_crop_by_pos_neg_labeld, num_samples=1))

    if transform_options.rand_flipd:
        training_transforms.append(transforms.RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50))

    if transform_options.rand_rotate_90d:
        training_transforms.append(transforms.RandRotate90d(keys=keys, prob=0.1, max_k=3))

    if transform_options.rand_scale_intensityd:
        training_transforms.append(transforms.RandScaleIntensityd(keys=keys[0], factors=0.1, prob=0.1))

    if transform_options.rand_shift_intensityd:
        training_transforms.append(transforms.RandShiftIntensityd(keys=keys[0], offsets=0.1, prob=0.1))

    if transform_options.grid_patchd is not None:
        training_transforms.append(transforms.GridPatchd(keys, patch_size=transform_options.grid_patchd))

    if transform_options.to_tensord:
        to_tensor = transforms.ToTensord(keys=keys[0], dtype=torch.float)
        training_transforms.append(to_tensor)
        validation_transforms.append(to_tensor)

    training_transform = transforms.Compose(training_transforms)
    testing_transform = transforms.Compose(validation_transforms)
    return training_transform, testing_transform


class ISegModality(Enum):
    T1 = 0
    T2 = 1

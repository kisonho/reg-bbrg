import glob, random as rand
from monai.data.dataset import CacheDataset, Dataset as MonaiDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import ToTensord
from torchmanager_core import os, torch, view
from torchmanager_core.typing import Enum


class RoundLable(MapTransform):
    def __init__(self, key):
        self.k = key
    def __call__(self, d):
        d = dict(d)
        d[self.k] = torch.round(d[self.k])
        return d


def load(train_image_dir: str, train_label_dir: str, img_size: int | tuple[int, ...], roi_size: int | tuple[int, ...] | None = None, train_split: int | None = None, validation_image_dir: str | None = None, validation_label_dir: str | None = None, for_testing: bool = False, show_verbose: bool = False, ndim: int = 3, search_key: str = '*.nii.gz', chached: bool = False, cache_num: int | tuple[int, ...] = 1, num_workers: int = 0, logger: view.logging.Logger | None = None) -> tuple[MonaiDataset, MonaiDataset]:
    # initialize logger
    if logger:
        logger.info(f'Loading images from single volume from source {train_image_dir}...')

    # index 0 is the image volume, index 1 is the ground truth
    keys = ['image', 'label']

    # Load the training images/labels in a dictionary
    train_images = sorted(glob.glob(os.path.join(train_image_dir, search_key)))
    train_labels = sorted(glob.glob(os.path.join(train_label_dir, search_key)))
    img_size = img_size if isinstance(img_size, tuple) else tuple([img_size for _ in range(ndim)])
    roi_size = roi_size if isinstance(roi_size, tuple) else tuple([roi_size for _ in range(ndim)]) if roi_size is not None else None
    train_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(train_images, train_labels)]

    # If this is for training, shuffled the list for randomness
    if not for_testing:
        rand.shuffle(train_dict)
    else:
        img_size = (img_size[0], img_size[1], -1)
        train_dict = train_dict

    # check if a validation directory is defined
    if (validation_image_dir and validation_label_dir):
        # If a validation data directory is defined, create a dictionary for them
        val_images = sorted(glob.glob(os.path.join(validation_image_dir, search_key)))
        val_labels = sorted(glob.glob(os.path.join(validation_label_dir, search_key)))
        val_dict = [{keys[0]: img, keys[1]: lab} for img, lab in zip(val_images, val_labels)]
    else:
        # If no validation dir is defined, split the training one into two as defined by train_split
        if not isinstance(train_split, int): raise ValueError(f'If a validation image and label directory are not specified, use train_split to create a validation set from the trianing cohort wtih n=train_split cases')
        val_dict = train_dict[-train_split:]
        train_dict = train_dict[:-train_split]

    # Creating the sequence of transformations for validation
    validation_transform_list: list[MapTransform] = [
        LoadImaged(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=img_size, num_samples=1),
        # CropForegroundd(keys=keys, source_key=keys[0]),
        # NormalizeIntensityd(keys=keys[0], channel_wise=True),
    ]

    # add random crop if roi size is defined
    training_transform_list: list[MapTransform] = [
        RandCropByPosNegLabeld(keys=keys, image_key='image', label_key='label', neg=0, spatial_size=roi_size, num_samples=1),
    ] if roi_size is not None else []
    training_transform_list = validation_transform_list + training_transform_list

    # replace validation transforms with training transforms if not for testing
    if not for_testing:
        validation_transform_list = training_transform_list
    # else: validation_transform_list += [Resized(keys=keys, spatial_size=(64, 64, -1))]

    # Rounding because of floating points
    training_transform_list.append(RoundLable(key='label'))
    validation_transform_list.append(RoundLable(key='label'))

    # Turning into tensors
    training_transform_list.append(ToTensord(keys=keys))
    validation_transform_list.append(ToTensord(keys=keys))

    # wrap to compose
    train_transforms = Compose(training_transform_list)
    val_transforms = Compose(validation_transform_list)
    cache_num = cache_num if isinstance(cache_num, tuple) else tuple([cache_num, cache_num])

    # load dataset
    if not chached:
        train_dataset = MonaiDataset(data=train_dict, transform=train_transforms,)
        val_dataset = MonaiDataset(data=val_dict, transform=val_transforms,)
    else:
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_num=cache_num[0], num_workers=num_workers, cache_rate=1.0, progress=show_verbose)
        val_dataset = CacheDataset(data=val_dict, transform=val_transforms, cache_num=cache_num[1], num_workers=num_workers, cache_rate=1.0, progress=show_verbose)
    return train_dataset, val_dataset


class BraTSModality(Enum):
    FLAIR = 0
    T1 = 1
    T1GD = 2
    T2 = 3

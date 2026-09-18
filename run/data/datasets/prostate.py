import glob, os, random as rand
from monai.data.dataset import CacheDataset, Dataset as MonaiDataset
from monai.transforms.compose import Compose
from monai.transforms.croppad.dictionary import SpatialPadd, RandSpatialCropSamplesd
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd, RandFlipd, RandRotate90d
from monai.transforms.transform import MapTransform
from monai.transforms.utility.dictionary import ToTensord

def load(
        train_image_dir: str, 
        img_size: int | tuple[int, ...], 
        roi_size: int | tuple[int, ...] | None = None,
        for_testing: bool = False, 
        spatial_size: tuple[int, ...] = (256, 256, -1),
        show_verbose: bool = False, 
        ndim: int = 3, 
        search_key: str = '*.nii.gz', 
        chached: bool = False, 
        cache_num: int | tuple[int, ...] = 1, 
        num_workers: int = 0
        ) -> tuple[MonaiDataset, MonaiDataset]:
    # Load the training images/labels in a dictionary
    keys = ['image']
    train_images = sorted(glob.glob(os.path.join(train_image_dir, search_key)))
    print(f"{len(train_images)=}")
    img_size = img_size if isinstance(img_size, tuple) else tuple([img_size for _ in range(ndim)])
    roi_size = roi_size if isinstance(roi_size, tuple) else tuple([roi_size for _ in range(ndim)]) if roi_size is not None else None
    train_dict = [{keys[0]: img} for img in train_images]

    # If this is for training, shuffled the list for randomness
    if not for_testing:
        rand.shuffle(train_dict)
    else:
        img_size = (img_size[0], img_size[1], -1)

    # Creating the sequence of transformations for validation
    validation_transform_list: list[MapTransform] = [
        LoadImaged(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        SpatialPadd(keys=keys, spatial_size=spatial_size, mode='minimum'),
        RandSpatialCropSamplesd(keys=keys, roi_size=img_size, num_samples=1, random_center=False, random_size=False),
    ]

    # add random crop if roi size is defined
    training_transform_list = [
        RandFlipd(keys=keys, spatial_axis=[i for i in range(3)], prob=0.50),
        RandRotate90d(keys=keys, prob=0.5, max_k=3),
        ]
    training_transform_list = validation_transform_list + training_transform_list

    # replace validation transforms with training transforms if not for testing
    if not for_testing:
        validation_transform_list = training_transform_list
    # if NOT for training, make the "train" set use validation datasets
    else: 
        training_transform_list = validation_transform_list

    # Turning into tensors
    training_transform_list.append(ToTensord(keys=keys))
    validation_transform_list.append(ToTensord(keys=keys))

    # wrap to compose
    train_transforms = Compose(training_transform_list)
    cache_num = cache_num if isinstance(cache_num, tuple) else tuple([cache_num, cache_num])

    # load dataset
    if not chached:
        train_dataset = MonaiDataset(data=train_dict, transform=train_transforms,)
    else:
        train_dataset = CacheDataset(data=train_dict, transform=train_transforms, cache_num=cache_num[0], num_workers=num_workers, cache_rate=1.0, progress=show_verbose)
    return train_dataset, train_dataset

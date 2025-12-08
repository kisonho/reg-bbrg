import itk
from torchmanager_core import devices, os, torch

from reg_bbrg.data import ResizeMode, MedicalTranslationDataset
from .datasets import BraTSModality, ImageType, ISeg, ISegModality, ISegTransformOptions, SupportedMedicalDatasets, load_brats, load_iseg_transforms, load_ixi, load_prostate

torch.multiprocessing.set_sharing_strategy('file_system')


def load_medical(dataset: SupportedMedicalDatasets, root_dir: str, /, batch_size: int, input_modality: int | list[int] = 0, target_modality: int = 1, *, device: torch.device | None = None, random_sampling: bool = True, train_split: int | None = None) -> tuple[MedicalTranslationDataset, MedicalTranslationDataset, MedicalTranslationDataset, int, int]:
    """
    Load the dataset.

    - Parameters:
        - root_dir: The root directory of the dataset in `str`.
        - batch_size: The batch size in `int`.
        - input_modality: The index or index list of input modalities in `int`
        - target_modality: The index or index list of target modalities in `int`
        - device: The device to load the data on in `torch.device`.
        - random_sampling: A `bool` flag of if randomly sampling the volume slices into batch size
        - train_split: An `int` of training split number
    - Returns: A `tuple` of the training, validation, and testing dataset in `MedicalTranslationDataset`, along with the number of input channels and output channels in `int`.
    """
    # initialize device
    num_workers = os.cpu_count()
    num_workers = 0 if num_workers is None else num_workers
    device = devices.CPU if device is None else device

    # check supported datasets
    match dataset:
        case SupportedMedicalDatasets.BRATS:
            # initialize dataset directories
            training_image_dir = os.path.join(root_dir, "Images.Training")
            training_label_dir = os.path.join(root_dir, "Labels.Training")
            testing_image_dir = os.path.join(root_dir, "Images.Testing")
            testing_label_dir = os.path.join(root_dir, "Labels.Testing")

            # initialize image size for BraTS, slice size as batch size
            if random_sampling:
                img_size = (240, 240, batch_size)
                b = 1
            else:
                img_size = (240, 240, 155)
                b = batch_size

            # limit modality index
            input_modality = BraTSModality(input_modality).value
            target_modality = BraTSModality(target_modality).value

            # load training and validation datasets
            training_dataset, validation_dataset = load_brats(training_image_dir, training_label_dir, img_size, train_split=train_split, chached=True, cache_num=(10, 10), num_workers=num_workers)
            training_dataset = MedicalTranslationDataset(training_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)
            validation_dataset = MedicalTranslationDataset(validation_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)

            # reinitialize image size for BraTS with full slices
            img_size = (240, 240, 155)

            # load testing dataset
            _, testing_dataset = load_brats(testing_image_dir, testing_label_dir, img_size, train_split = 0, for_testing=True, chached=True, cache_num=(10, 10), num_workers=num_workers)
            print(len(testing_dataset))
            testing_dataset = MedicalTranslationDataset(testing_dataset, batch_size, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=False, num_workers=num_workers, use_slices=True)#, slice_size=batch_size)

            # channels information
            input_channels = output_channels = 1
        case SupportedMedicalDatasets.ISEG:
            # initialize
            itk.ProcessObject.SetGlobalWarningDisplay(False) # type: ignore
            keys = ['image', 'label']

            # initalize image size for iseg, slice size as batch size
            if random_sampling:
                img_size = (-1, -1, batch_size)
                b = 1
            else:
                img_size = (256, 256, -1)
                b = batch_size

            # limit modality index
            input_modality = ISegModality(input_modality).value
            target_modality = ISegModality(target_modality).value

            # load transforms
            transform_options = ISegTransformOptions(
                load_imaged=False,
                normalize_intensityd=False,
                crop_foregroundd=True,
                orientationd=True,
                rand_crop_by_pos_neg_labeld=img_size,
                rand_flipd=True,
                rand_scale_intensityd=True,
                rand_shift_intensityd=True,
                to_tensord=True,
            )
            training_transform, testing_transform = load_iseg_transforms(transform_options, keys)

            # load dataset
            train_split = 1 if train_split is None else train_split
            train_val_dataset = ISeg(root_dir, data_prefix=[f"subject-{i}" for i in range(1, 8)], shuffle=True, transform=training_transform, type=ImageType.IMG)
            training_dataset, validation_dataset = train_val_dataset.split(7 - train_split)

            # resize to 256x256 for each slice
            img_size = (256, 256, -1)

            # wrap to translation dataset
            training_dataset = MedicalTranslationDataset(training_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, resize_mode=ResizeMode.RESIZE, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)
            validation_dataset = MedicalTranslationDataset(validation_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, resize_mode=ResizeMode.RESIZE, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)

            # reinitialize image size for iseg with full slices for testing
            img_size = (-1, -1, 144)

            # load transforms for testing
            transform_options = ISegTransformOptions(
                load_imaged=False,
                normalize_intensityd=False,
                crop_foregroundd=True,
                orientationd=True,
                spatial_padd=img_size,
                to_tensord=True,
            )
            _, testing_transform = load_iseg_transforms(transform_options, keys)

            # resize to 256x256x144
            img_size = (256, 256, 144)

            # load testing dataset
            testing_dataset = ISeg(root_dir, data_prefix=[f"subject-{i}" for i in range(8, 11)], transform=testing_transform, type=ImageType.IMG)
            testing_dataset = MedicalTranslationDataset(testing_dataset, batch_size, img_size, input_modality=input_modality, target_modality=target_modality, resize_mode=ResizeMode.RESIZE, shuffle=False, num_workers=num_workers, use_slices=True)

            # channels information
            input_channels = output_channels = 1
        case SupportedMedicalDatasets.PROSTATE:
            # initialize dataset directories
            training_image_dir = os.path.join(root_dir, "training")
            testing_image_dir = os.path.join(root_dir, "testing")
            assert train_split is None, "Training split is not supported for ProstateMRI dataset."

            # initialize dimensions
            if random_sampling:
                img_size = (256, 256, batch_size)
                b = 1
            else:
                img_size = (256, 256, 58)
                b = batch_size

            # load training and validation datasets
            training_dataset, _ = load_prostate(training_image_dir, img_size, chached=True, cache_num=(10, 10), num_workers=num_workers)
            training_dataset = MedicalTranslationDataset(training_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)

            # reinitialize image size for full slices
            img_size = (256, 256, 58)

            # load testing dataset
            testing_dataset, _ = load_prostate(testing_image_dir, img_size, for_testing=True, spatial_size=img_size, chached=True, cache_num=(10, 10), num_workers=num_workers)
            validation_dataset = testing_dataset = MedicalTranslationDataset(testing_dataset, batch_size, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=False, num_workers=num_workers, use_slices=True)

            # channels information
            input_channels = 2
            output_channels = 1
        case SupportedMedicalDatasets.IXI:
            try:
                import resource
                soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                target_limit = min(hard_limit, max(soft_limit, 4096))
                if soft_limit < target_limit:
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
            except Exception:
                pass

            # initialize dataset directories
            t1_image_dir = os.path.join(root_dir, "IXI-T1")
            t2_image_dir = os.path.join(root_dir, "IXI-T2")
            assert train_split is None, "Training split is not supported for ProstateMRI dataset."

            # initialize dimensions
            if random_sampling:
                img_size = (256, 256, batch_size)
                b = 1
            else:
                img_size = (256, 256, 150)
                b = batch_size

            # load datasets
            training_dataset, testing_dataset = load_ixi(t1_image_dir, t2_image_dir, cache_rate=0.1, img_size=img_size, random_crop=random_sampling)
            training_dataset = MedicalTranslationDataset(training_dataset, b, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=True, num_workers=num_workers, use_slices=not random_sampling)

            # reinitialize image size for full resolution
            img_size = (256, 256, 150)

            # load testing dataset
            validation_dataset = testing_dataset = MedicalTranslationDataset(testing_dataset, batch_size, img_size, input_modality=input_modality, target_modality=target_modality, device=device, shuffle=False, num_workers=num_workers, use_slices=True)

            # channels information
            input_channels = output_channels = 1
    return training_dataset, validation_dataset, testing_dataset, input_channels, output_channels

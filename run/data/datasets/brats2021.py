"""BraTS 2021 subject folders, using the BraTS 2018 modality order."""
import torch
from monai.data.dataset import CacheDataset, Dataset
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import Orientationd
from monai.transforms.croppad.dictionary import RandCropByPosNegLabeld
from monai.transforms.utility.dictionary import ToTensord
from pathlib import Path

from reg_bbrg.data import MedicalTranslationDataset
from reg_bbrg.data.protocols import MedicalTranslationData

Subject = dict[str, list[str] | str]


def discover_subjects(root_dir: str) -> list[Subject]:
    """Index subject folders and reject missing or ambiguous modality files."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"BraTS 2021 directory does not exist: {root}")
    subjects = sorted(path for path in root.glob("BraTS2021_*") if path.is_dir())
    if not subjects:
        raise ValueError(f"No BraTS2021_* subject folders found in {root}")
    records = []
    for subject in subjects:
        files = {}
        for modality in ("flair", "t1", "t1ce", "t2", "seg"):
            candidates = [subject / f"{subject.name}_{modality}{suffix}" for suffix in (".nii", ".nii.gz")]
            matches = [path for path in candidates if path.is_file()]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {modality} NIfTI file for {subject.name}; found {len(matches)} in {subject}")
            files[modality] = str(matches[0])
        records.append({"image": [files[key] for key in ("flair", "t1", "t1ce", "t2")], "label": files["seg"]})
    return records


def discover_splits(root_dir: str) -> tuple[list[Subject], list[Subject], list[Subject]]:
    """Read existing Train/Val/Test folders without repartitioning subjects."""
    splits = []
    seen: dict[str, str] = {}
    for name in ("Train", "Val", "Test"):
        records = discover_subjects(str(Path(root_dir).expanduser() / name))
        for record in records:
            subject = Path(str(record["label"])).parent.name
            if subject in seen:
                raise ValueError(f"Subject {subject} appears in both {seen[subject]} and {name}.")
            seen[subject] = name
        splits.append(records)
    return splits[0], splits[1], splits[2]


def build_dataset(records: list[Subject], img_size: tuple[int, int, int], *, for_testing: bool = False, cache_num: int = 10, num_workers: int = 0) -> Dataset:
    """Stack modalities as C,H,W,D and apply the existing BraTS crop policy."""

    keys = ["image", "label"]
    spatial_size = (img_size[0], img_size[1], -1) if for_testing else img_size
    transform = Compose([
        LoadImaged(keys=keys, ensure_channel_first=True),
        Orientationd(keys=keys, axcodes="RAS"),
        RandCropByPosNegLabeld(keys=keys, image_key="image", label_key="label", neg=0, spatial_size=spatial_size, num_samples=1),
        ToTensord(keys=keys),
    ])
    if cache_num == 0:
        return Dataset(data=records, transform=transform)
    return CacheDataset(data=records, transform=transform, cache_num=cache_num, num_workers=num_workers, progress=False)


def load(root_dir: str, img_size: tuple[int, int, int], *, cache_num: int = 10, num_workers: int = 0) -> tuple[Dataset, Dataset, Dataset]:
    """Load the user-provided Train/Val/Test partition from one dataset root."""
    training, validation, testing = discover_splits(root_dir)
    return (
        build_dataset(training, img_size, cache_num=cache_num, num_workers=num_workers),
        build_dataset(validation, img_size, cache_num=cache_num, num_workers=num_workers),
        build_dataset(testing, img_size, for_testing=True, cache_num=cache_num, num_workers=num_workers),
    )


class BraTS2021TranslationDataset(MedicalTranslationDataset):
    """Accept the collated list of crops produced by the BraTS 2021 loader."""

    @staticmethod
    def unpack_data(batch: MedicalTranslationData | list[MedicalTranslationData]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return MedicalTranslationDataset.unpack_data(batch)
        if not batch:
            raise ValueError("Cannot unpack an empty list of BraTS 2021 crops.")

        # Each crop contains [batch, slices, channels, ...] tensors.
        pairs = [MedicalTranslationDataset.unpack_data(crop) for crop in batch]
        return pairs[0] if len(pairs) == 1 else (torch.cat([pair[0] for pair in pairs], dim=0), torch.cat([pair[1] for pair in pairs], dim=0))

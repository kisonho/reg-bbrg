import os, re, glob, torch
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.utility.dictionary import EnsureChannelFirstd, EnsureTyped
from monai.transforms.spatial.dictionary import Orientationd, ResampleToMatchd, Resized, Spacingd
from monai.transforms.intensity.dictionary import ScaleIntensityd
from monai.transforms.croppad.dictionary import RandSpatialCropd, CropForegroundd
from monai.data.dataset import CacheDataset
from reg_bbrg.data import MonaiData
from typing import Sequence, cast

_IXI_ID_RE = re.compile(r"(IXI\d+)", re.IGNORECASE)
_NIFTI_GLOBS = ("*.nii", "*.nii.gz")

def _list_nii(root: str, patterns: Sequence[str] = _NIFTI_GLOBS) -> list[str]:
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    return sorted(set(files))

def _extract_subject_id(path: str) -> str | None:
    for part in os.path.normpath(path).split(os.sep)[::-1]:
        m = _IXI_ID_RE.search(part)
        if m:
            return m.group(1).upper()
    m = _IXI_ID_RE.search(os.path.basename(path))
    return m.group(1).upper() if m else None

def _build_paired_ixi_dicts(t1_root: str, t2_root: str) -> list[dict[str, str]]:
    t1_files = _list_nii(t1_root)
    t2_files = _list_nii(t2_root)
    id2t1: dict[str, str] = {}
    id2t2: dict[str, str] = {}
    for f in t1_files:
        sid = _extract_subject_id(f)
        if sid: id2t1[sid] = f
    for f in t2_files:
        sid = _extract_subject_id(f)
        if sid: id2t2[sid] = f
    common = sorted(set(id2t1) & set(id2t2))
    data = [{"input": id2t1[s], "target": id2t2[s]} for s in common]
    if not data:
        raise RuntimeError(f"No paired subjects found in '{t1_root}' and '{t2_root}'.")
    return data


class TensorPairCacheDataset(CacheDataset):
    """CacheDataset returning `(input, target)` tensor pairs instead of a dictionary."""

    def __getitem__(self, index: int) -> MonaiData:
        item = super().__getitem__(index)
        if isinstance(item, dict):
            if "input" not in item or "target" not in item:
                raise KeyError("IXI dataset samples must contain 'input' and 'target' keys.")
            input_tensor = item["input"]
            target_tensor = item["target"]
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            input_tensor, target_tensor = item
        else:
            raise TypeError(f"Unsupported IXI sample type: {type(item)!r}")

        input_tensor = cast(torch.Tensor, input_tensor) if torch.is_tensor(input_tensor) else torch.as_tensor(input_tensor)
        target_tensor = cast(torch.Tensor, target_tensor) if torch.is_tensor(target_tensor) else torch.as_tensor(target_tensor)
        img = torch.cat([input_tensor, target_tensor])
        img = img * 2 - 1
        return MonaiData(image=img)


# ---------- Volume preprocessing (3D, once) ----------
def load_ixi_transforms(spacing: tuple[float,float,float] | None = None, img_size: tuple[int, ...] | None = None, random_crop: bool = False) -> Compose:
    keys = ("input", "target")
    tx = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys), # -> (C, D, H, W)
        Orientationd(keys=keys, axcodes="RAS"),
        # CropForegroundd(keys="input", source_key="input"),
        # CropForegroundd(keys="target", source_key="target"),
        ResampleToMatchd(keys=["input", "target"], key_dst="target", mode="bilinear"),
    ]
    if spacing is not None:
        tx.append(Spacingd(keys=keys, pixdim=spacing, mode=("bilinear","bilinear")))
    tx += [
        ScaleIntensityd(keys=keys),
        EnsureTyped(keys=keys, data_type="tensor"),
    ]
    if random_crop:
        assert img_size is not None, "Image size must be given."
        random_crop_size = (-1, -1, img_size[-1])
        tx.append(RandSpatialCropd(keys=keys, roi_size=random_crop_size))
    if img_size:
        tx.append(Resized(keys=keys, spatial_size=img_size, mode="trilinear"))
    return Compose(tx)


def load_ixi(
    t1_root: str,
    t2_root: str,
    /,
    img_size: tuple[int, ...] | None = None,
    *,
    spacing: tuple[float,float,float] | None = (1.0, 1.0, 1.0),
    cache_rate: float = 1.0,
    num_workers: int | None = os.cpu_count(),
    random_crop: bool = False,
    val_frac: float = 0.3,
    seed: int = 0,
) -> tuple[TensorPairCacheDataset, TensorPairCacheDataset]:
    """Returns paired slice datasets expanded from cached volumes."""
    cpu_count = max(1, os.cpu_count() or 1)
    torch_threads = max(1, torch.get_num_threads())
    if num_workers is None:
        num_workers = cpu_count
    num_workers = max(0, num_workers)
    if num_workers:
        max_workers = max(1, cpu_count // torch_threads)
        num_workers = min(num_workers, max_workers)
        torch.set_num_threads(max(1, min(torch_threads, cpu_count // num_workers)))
    else:
        torch.set_num_threads(min(torch_threads, cpu_count))
    # 1) discover subjects
    data = _build_paired_ixi_dicts(t1_root, t2_root)

    # 2) split by subject (deterministic)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(data), generator=g).tolist()
    n_val = max(1, int(round(len(data)*val_frac)))
    val_ids = set(perm[:n_val])
    train_data = [data[i] for i in range(len(data)) if i not in val_ids]
    val_data   = [data[i] for i in val_ids]

    # 3) preprocess/cache volumes
    training_transform = load_ixi_transforms(spacing, img_size=img_size, random_crop=random_crop)
    testing_transform = load_ixi_transforms(spacing, img_size=img_size)
    train_vols = TensorPairCacheDataset(train_data, training_transform, cache_rate=cache_rate, num_workers=num_workers)
    val_vols = TensorPairCacheDataset(val_data, testing_transform, cache_rate=cache_rate, num_workers=num_workers)
    return train_vols, val_vols

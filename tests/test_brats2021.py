"""Subject indexing tests run without the optional training dependencies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("brats2021", Path(__file__).resolve().parents[1] / "example/data/datasets/brats2021.py")
brats2021 = importlib.util.module_from_spec(spec)
HAS_MONAI = importlib.util.find_spec("monai") is not None
if HAS_MONAI:
    spec.loader.exec_module(brats2021)
else:
    # Indexing tests need no imaging runtime; the real transform test is skipped.
    modules = {name: ModuleType(name) for name in ("monai", "monai.data", "monai.transforms")}
    for name in ("CacheDataset", "Dataset"):
        setattr(modules["monai.data"], name, object)
    for name in ("Compose", "LoadImaged", "Orientationd", "RandCropByPosNegLabeld", "ToTensord"):
        setattr(modules["monai.transforms"], name, object)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(brats2021)


class TestBraTS2021(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def subject(self, name="BraTS2021_01165", split=None):
        directory = (self.root / split if split else self.root) / name
        directory.mkdir(parents=True)
        for index, modality in enumerate(("flair", "t1", "t1ce", "t2", "seg")):
            suffix = ".nii.gz" if index % 2 else ".nii"
            (directory / f"{name}_{modality}{suffix}").touch()
        return directory

    def test_modality_order_and_mixed_extensions(self):
        self.subject("BraTS2021_01165")
        self.subject("BraTS2021_00001")
        (self.root / "unrelated").mkdir()
        records = brats2021.discover_subjects(str(self.root))
        self.assertEqual(len(records), 2)
        self.assertEqual([Path(p).name for p in records[0]["image"]], [
            "BraTS2021_00001_flair.nii", "BraTS2021_00001_t1.nii.gz",
            "BraTS2021_00001_t1ce.nii", "BraTS2021_00001_t2.nii.gz",
        ])
        self.assertEqual(Path(records[0]["label"]).name, "BraTS2021_00001_seg.nii")

    def test_missing_modality_rejected(self):
        directory = self.subject()
        (directory / "BraTS2021_01165_t1ce.nii").unlink()
        with self.assertRaisesRegex(ValueError, "t1ce.*BraTS2021_01165"):
            brats2021.discover_subjects(str(self.root))

    def test_duplicate_extension_rejected(self):
        directory = self.subject()
        (directory / "BraTS2021_01165_seg.nii.gz").touch()
        with self.assertRaisesRegex(ValueError, "seg.*found 2"):
            brats2021.discover_subjects(str(self.root))

    def test_empty_or_missing_root_rejected(self):
        with self.assertRaises(ValueError):
            brats2021.discover_subjects(str(self.root))
        with self.assertRaises(FileNotFoundError):
            brats2021.discover_subjects(str(self.root / "missing"))

    def test_existing_assignments_preserved_with_nonconsecutive_ids(self):
        self.subject("BraTS2021_01165", "Train")
        self.subject("BraTS2021_00007", "Train")
        self.subject("BraTS2021_00001", "Val")
        self.subject("BraTS2021_00099", "Test")
        splits = brats2021.discover_splits(str(self.root))
        ids = [[Path(record["label"]).parent.name for record in part] for part in splits]
        self.assertEqual(ids, [["BraTS2021_00007", "BraTS2021_01165"], ["BraTS2021_00001"], ["BraTS2021_00099"]])
        self.assertEqual(splits, brats2021.discover_splits(str(self.root)))

    def test_missing_or_empty_split_rejected(self):
        self.subject("BraTS2021_00007", "Train")
        with self.assertRaisesRegex(FileNotFoundError, "Val"):
            brats2021.discover_splits(str(self.root))
        (self.root / "Val").mkdir()
        with self.assertRaisesRegex(ValueError, "Val"):
            brats2021.discover_splits(str(self.root))

    def test_duplicate_subject_across_splits_rejected(self):
        self.subject("BraTS2021_00007", "Train")
        self.subject("BraTS2021_00007", "Val")
        self.subject("BraTS2021_00099", "Test")
        with self.assertRaisesRegex(ValueError, "BraTS2021_00007.*Train.*Val"):
            brats2021.discover_splits(str(self.root))

    def test_load_uses_each_folder_and_full_depth_only_for_test(self):
        for name, subject in (("Train", "00007"), ("Val", "00001"), ("Test", "00099")):
            self.subject(f"BraTS2021_{subject}", name)
        with patch.object(brats2021, "build_dataset", side_effect=lambda records, *args, **kwargs: records) as build:
            splits = brats2021.load(str(self.root), (240, 240, 4), cache_num=0, num_workers=2)
        self.assertEqual(splits, brats2021.discover_splits(str(self.root)))
        self.assertEqual(build.call_count, 3)
        for index, call in enumerate(build.call_args_list):
            self.assertEqual(call.args, (splits[index], (240, 240, 4)))
            self.assertEqual(call.kwargs.get("for_testing", False), index == 2)
            self.assertEqual(call.kwargs["cache_num"], 0)
            self.assertEqual(call.kwargs["num_workers"], 2)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("monai", "nibabel", "torch")), "Requires MONAI, nibabel and PyTorch")
    def test_nifti_stacking_crop_and_test_depth(self):
        import nibabel as nib
        import numpy as np

        self.subject()
        records = brats2021.discover_subjects(str(self.root))
        for index, path in enumerate(records[0]["image"]):
            nib.save(nib.Nifti1Image(np.full((8, 8, 6), index + 1, dtype=np.float32), np.eye(4)), path)
        nib.save(nib.Nifti1Image(np.ones((8, 8, 6), dtype=np.int16), np.eye(4)), records[0]["label"])
        for cache_num in (0, 1):
            with self.subTest(cache_num=cache_num):
                train = brats2021.build_dataset(records, (8, 8, 2), cache_num=cache_num)[0][0]
                test = brats2021.build_dataset(records, (8, 8, 2), for_testing=True, cache_num=cache_num)[0][0]
                self.assertEqual(tuple(train["image"].shape), (4, 8, 8, 2))
                self.assertEqual(tuple(train["label"].shape), (1, 8, 8, 2))
                self.assertEqual(tuple(test["image"].shape), (4, 8, 8, 6))
                for index in range(4):
                    self.assertTrue((test["image"][index] == index + 1).all())


if __name__ == "__main__":
    unittest.main()

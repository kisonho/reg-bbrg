"""Exercise CLI and dataset wiring without loading the optional GPU runtime."""
import argparse
import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def definition(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


class DatasetName(Enum):
    BRATS2021 = "brats2021"
    BRATS2018 = "brats2018"


class Modality(Enum):
    FLAIR = 0
    T1 = 1
    T1CE = 2
    T2 = 3


class Model(Enum):
    A_BRIDGE = "a_bridge"


class BaseConfigs:
    def format_arguments(self):
        pass

    @staticmethod
    def get_arguments(parser):
        return parser


class TestBraTS2021Modalities(unittest.TestCase):
    def parse(self, arguments):
        cfgs = definition("run/__main__.py", "Configs", {
            "TrainingConfigs": BaseConfigs,
            "argparse": argparse,
            "data": SimpleNamespace(SupportedMedicalDatasets=DatasetName),
            "networks": SimpleNamespace(UNetType=Model),
            "cast": lambda annotation, value: value,
        })
        config = cfgs()
        vars(config).update(vars(cfgs.get_arguments(argparse.ArgumentParser()).parse_args(arguments)))
        config.format_arguments()
        return config

    def load(self, input_modality, target_modality=0):
        wrapper = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(args=args, **kwargs))
        loader = Mock(return_value=("train", "val", "test"))
        load = definition("run/data/loader.py", "load_medical", {
            "os": SimpleNamespace(cpu_count=lambda: 0),
            "devices": SimpleNamespace(CPU="cpu"),
            "SupportedMedicalDatasets": DatasetName,
            "BraTSModality": Modality,
            "BraTS2021TranslationDataset": wrapper,
            "load_brats2021": loader,
        })
        return load(DatasetName.BRATS2021, "data", batch_size=4, input_modality=input_modality, target_modality=target_modality)

    def test_t1_t2_to_flair_cli_and_all_splits(self):
        config = self.parse(["brats2021", "--input_modality", "1", "3", "--target_modality", "0"])
        self.assertEqual(config.input_modality, [1, 3])
        train, val, test, inputs, outputs = self.load(config.input_modality, config.target_modality)
        self.assertEqual((inputs, outputs), (2, 1))
        for dataset in (train, val, test):
            self.assertEqual(dataset.input_modality, [1, 3])
            self.assertEqual(dataset.target_modality, 0)
        self.assertFalse(train.use_slices)
        self.assertTrue(test.use_slices)

    def test_single_input_preserves_scalar_interface(self):
        for dataset in ("brats2018", "brats2021"):
            config = self.parse([dataset, "-input", "1", "-target", "3"])
            self.assertEqual(config.input_modality, 1)
        self.assertEqual(self.load(1, 3)[-2:], (1, 1))

    def test_order_is_preserved(self):
        self.assertEqual(self.load([3, 1])[0].input_modality, [3, 1])

    def test_bad_indices_duplicates_and_empty_inputs_rejected(self):
        for inputs in ([], [1, 1], [1, 9]):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                self.load(inputs)

    def test_multichannel_cli_rejected_for_other_datasets(self):
        with self.assertRaisesRegex(ValueError, "only for brats2021"):
            self.parse(["brats2018", "-input", "1", "3"])


if __name__ == "__main__":
    unittest.main()

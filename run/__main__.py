from reg_bbrg import networks
from reg_bbrg.configs import EvalConfigs, TrainingConfigs
from torchmanager_core import argparse, view
from torchmanager_core.typing import cast

from . import data
from .engine import eval, train


class Configs(TrainingConfigs):
    dataset: data.SupportedMedicalDatasets
    disable_ema: bool
    disable_random_sampling: bool
    input_modality: int | list[int]
    model_type: networks.UNetType
    target_modality: int
    train_split: int | None
    with_gen_time_emb: bool

    def format_arguments(self) -> None:
        super().format_arguments()
        # Preserve the scalar interface for existing single-input datasets.
        if isinstance(self.input_modality, list) and len(self.input_modality) == 1:
            self.input_modality = self.input_modality[0]
        if isinstance(self.input_modality, list) and self.dataset != data.SupportedMedicalDatasets.BRATS2021:
            raise ValueError("Multiple --input_modality values are currently supported only for brats2021.")
        assert self.train_split is None or self.train_split >= 0, "The train_split must be a non-negative value if given."

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("dataset", type=data.SupportedMedicalDatasets, choices=list(data.SupportedMedicalDatasets), help="The dataset to train.")
        parser.add_argument("-input", "--input_modality", type=int, nargs="+", default=0, help="Input modality index or indices (BraTS 2021: 1 3 for T1w + T2w).")
        parser.add_argument("-target", "--target_modality", type=int, default=1, help="The target modality to use.")
        parser.add_argument("--train_split", type=int, default=None, help="Dataset split count (validation count for BraTS 2018; omit for BraTS 2021's Train/Val/Test folders).")
        parser.add_argument("--disable_ema", action="store_true", default=False, help="A flag to disable EMA during training.")
        parser.add_argument("--disable_random_sampling", action="store_true", default=False, help="A flag to disable random sampling and use the full dataset.")
        parser.add_argument_group("Model arguments")
        parser.add_argument("-model", "--model_type", type=networks.UNetType, default=networks.UNetType.A_BRIDGE, choices=list(networks.UNetType), help="The model type to use.")
        parser.add_argument("--with_gen_time_emb", action="store_true", default=False, help="Whether to use generator time embedding.")
        parser = cast(argparse.ArgumentParser, TrainingConfigs.get_arguments(parser))
        return parser

    def show_settings(self) -> None:
        view.logger.info(f"Dataset {self.dataset}: input={self.input_modality}, target={self.target_modality}, random_sampling={not self.disable_random_sampling}")
        view.logger.info(f"Model: {self.model_type}, with_gen_time_emb={self.with_gen_time_emb}")
        super().show_settings()


if __name__ == "__main__":
    # load configs
    training_cfgs = cast(Configs, Configs.from_arguments())
    testing_cfgs = EvalConfigs.from_training(training_cfgs)

    # load datasets
    training_dataset, validation_dataset, testing_dataset, input_channels, output_channels = data.load_medical(training_cfgs.dataset, training_cfgs.data_dir, training_cfgs.batch_size, device=training_cfgs.devices[0] if training_cfgs.devices is not None else None, input_modality=training_cfgs.input_modality, target_modality=training_cfgs.target_modality, train_split=training_cfgs.train_split, random_sampling=not training_cfgs.disable_random_sampling)

    # load model
    model = networks.build(input_channels, output_channels, time_steps=training_cfgs.time_steps, model_type=training_cfgs.model_type, with_gen_time_emb=training_cfgs.with_gen_time_emb) if training_cfgs.ckpt_path is None else None

    # train
    train(training_cfgs, training_dataset, model, validation_dataset=validation_dataset if training_cfgs.dataset == data.SupportedMedicalDatasets.BRATS2021 else None)

    # evaluate
    result = eval(testing_cfgs, testing_dataset)
    view.logger.info(result)

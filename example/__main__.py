from reg_bbrg import networks
from reg_bbrg.configs import EvalConfigs, TrainingConfigs
from torchmanager_core import argparse, view
from torchmanager_core.typing import cast

from . import data
from .engine import eval, train


class Configs(TrainingConfigs):
    dataset: data.SupportedMedicalDatasets
    disable_random_sampling: bool
    input_modality: int
    model_type: networks.UNetType
    target_modality: int
    train_split: int | None
    with_gen_time_emb: bool

    def format_arguments(self) -> None:
        super().format_arguments()
        self.dataset = data.SupportedMedicalDatasets(self.dataset)
        self.model_type = networks.UNetType[self.model_type.upper()] if isinstance(self.model_type, str) else self.model_type
        assert self.train_split is None or self.train_split >= 0, "The train_split must be a non-negative value if given."

    @staticmethod
    def get_arguments(parser: argparse.ArgumentParser = argparse.ArgumentParser()) -> argparse.ArgumentParser:
        parser.add_argument("dataset", type=str, help="The dataset to train.")
        parser.add_argument("-input", "--input_modality", type=int, default=0, help="The input modality to use.")
        parser.add_argument("-target", "--target_modality", type=int, default=1, help="The target modality to use.")
        parser.add_argument("--train_split", type=int, default=None, help="The training script of the dataset")
        parser.add_argument("--disable_random_sampling", action="store_true", default=False, help="A flag to isable random sampling and use the full dataset.")

        parser.add_argument_group("Model arguments")
        parser.add_argument("-model", "--model_type", type=str, default="a_bridge", help="The model type to use.")
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
    train(training_cfgs, training_dataset, model)

    # evaluate
    result = eval(testing_cfgs, testing_dataset)
    view.logger.info(result)

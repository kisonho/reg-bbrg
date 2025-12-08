from argparse import _ArgumentGroup, ArgumentParser
from diffusion import VERSION as DIFF_VERSION
from diffusion.configs import TrainingConfigs as Configs
from torchmanager_core import view

from reg_bbrg.version import DESCRIPTION


class TrainingConfigs(Configs):
    lambda_rec: float
    lambda_adv: float

    def format_arguments(self) -> None:
        super().format_arguments()
        assert self.lambda_rec >= 0, 'Reconstruction loss weight must be non-negative.'
        assert self.lambda_adv >= 0, 'Discriminator loss weight must be non-negative.'

    @staticmethod
    def get_arguments(parser: ArgumentParser | _ArgumentGroup = ArgumentParser()) -> ArgumentParser | _ArgumentGroup:
        parser = Configs.get_arguments(parser)
        parser.add_argument_group("GAN-Guided Training Configurations")
        parser.add_argument("-rec", "--lambda_rec", type=float, default=1, help="The weight for the reconstruction loss.")
        parser.add_argument("-adv", "--lambda_adv", type=float, default=1, help="The weight for the adversarial loss.")
        return parser

    def show_settings(self) -> None:
        super().show_settings()
        view.logger.info(f"Regularization settings: lambda_rec={self.lambda_rec}, lambda_adv={self.lambda_adv}")

    def show_environments(self, description: str = DESCRIPTION) -> None:
        super().show_environments(description)
        view.logger.info(f"torchmanager-diffusion={DIFF_VERSION}")

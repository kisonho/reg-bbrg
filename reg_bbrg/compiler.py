from torchmanager import losses
from torchmanager_core import torch, _raise
from torchmanager_core.typing import TypeVar, cast

try:
    from diffusion.optim import EMAOptimizer
except ImportError:
    EMAOptimizer = None

from .managers import AdversarialDiffusionManager as Manager
from .nn import RegBBrgModule

G1 = TypeVar("G1", bound=torch.nn.Module)
G2 = TypeVar("G2", bound=torch.nn.Module)
D = TypeVar("D", bound=torch.nn.Module)


def compile(model: RegBBrgModule[G1, G2, D], /, lr: float = 2e-5, adv_lr: float = 2e-4, *, lambda_rec: float = 1, lambda_adv: float = 1, use_ema: bool = True) -> Manager[RegBBrgModule[G1, G2, D]]:
    """
    Compile the model and discriminator for adversarial training.

    - Parameters:
        - model: The model to compile in `torch.nn.Module`.
        - discriminator: The discriminator to compile in `torch.nn.Module`.
        - lr: The learning rate for the model in `float`.
        - adv_lr: The learning rate for the discriminator in `float`.
        - lambda_rec: The weight for the reconstruction loss in `float`.
        - lambda_adv: The weight for the adversarial loss in `float`.
        - use_ema: A `bool` flag of if using EMA optimizer
    - Returns: The compiled manager in `c2b2.managers.AdversarialDiffusionManager`.
    """
    # load generator optimizer and loss
    trainable_params = list(model.generator_parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    if use_ema:
        assert EMAOptimizer is not None, _raise(NotImplementedError("EMA optimizer is not implemented, torchmanager-diffusion v1.2 is required."))
        optimizer = EMAOptimizer(optimizer, trainable_params)
    loss_fn = {
        "diffusion": losses.MAE(target="target"),
        "rec": losses.MAE(target="rec", weight=lambda_rec),
        "adv_false": losses.Loss(torch.nn.BCEWithLogitsLoss(), target="d_false", weight=lambda_adv),
    }

    # load discriminator optimizer and loss
    adv_optimizer = torch.optim.Adam(model.discriminator_parameters(), lr=adv_lr)
    adv_loss_fn = cast(dict[str, losses.Loss], {
        "adv_true": losses.Loss(torch.nn.BCEWithLogitsLoss()),
    })

    # initialize manager
    return Manager(model, optimizer, loss_fn, adversarial_optimizer=adv_optimizer, adversarial_loss_fn=adv_loss_fn)

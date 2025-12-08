from diffusion import DiffusionData, Manager
from torch.nn.utils import clip_grad
from torch.optim.optimizer import Optimizer
from torchmanager.losses import Loss, MultiLosses, ParallelLoss
from torchmanager.metrics import Metric
from torchmanager_core import devices, torch
from torchmanager_core.typing import Callable, TypeVar, cast, overload

from .protocols import RegBBrgModule, RegBBrgOutput, PredictionContext

M = TypeVar("M", bound=RegBBrgModule)


class AdversarialDiffusionManager(Manager[M]):
    """
    The manager for adversarial training with conditional GAN model.
    
    - Parameters:
        - adversarial_optimizer: An optional `torch.optim.Optimizer` for adversarial training
        - adversarial_loss_fn: An optional `torchmanager.losses.Loss` function for adversarial training
        - compiled_adv_optimizer: The compiled adversarial optimizer in `torch.optim.Optimizer`
        - compiled_adv_loss_fn: The compiled adversarial loss function in `torchmanager.losses.Loss`
        - enable_adversarial_learning: Whether to enable adversarial learning
        - raw_adv_loss_fn: The `torchmanager.losses.Loss` controlled by this manager without `torch.nn.DataParallel` wrap
    """
    adversarial_optimizer: Optimizer | None
    """The adversarial optimizer for adversarial training"""
    adversarial_loss_fn: Loss | None
    """The adversarial loss function for adversarial training"""

    @property
    def compiled_adv_optimizer(self) -> Optimizer:
        """The compiled adversarial optimizer in `torch.optim.Optimizer`"""
        if self.adversarial_optimizer is None:
            raise ValueError("adversarial optimizer is not compiled.")
        return self.adversarial_optimizer

    @property
    def compiled_adv_loss_fn(self) -> Loss | Callable[..., None]:
        if self.adversarial_loss_fn is None:
            return lambda *_: None
        return self.adversarial_loss_fn

    @property
    def enable_adversarial_learning(self) -> bool:
        """Whether to enable adversarial learning"""
        return self.adversarial_optimizer is not None and self.adversarial_loss_fn is not None

    @property
    def raw_adv_loss_fn(self) -> Loss | None:
        """The `torchmanager.losses.Loss` controlled by this manager without `torch.nn.DataParallel` wrap"""
        return self.adversarial_loss_fn.module if isinstance(self.adversarial_loss_fn, ParallelLoss) else self.adversarial_loss_fn

    @property
    def return_prediction(self) -> bool:
        """Whether to return prediction in output"""
        return self.raw_model.return_prediction

    def __init__(self, model: M, optimizer: Optimizer | None = None, loss_fn: Loss | dict[str, Loss] | None = None, metrics: dict[str, Metric] = {}, *, adversarial_optimizer: Optimizer | None = None, adversarial_loss_fn: Loss | dict[str, Loss] | None = None) -> None:
        # initialize adversarial loss
        if isinstance(adversarial_loss_fn, dict):
            loss_fn_mapping: dict[str, Loss] = {f"{name}_loss": fn for name, fn in adversarial_loss_fn.items()}
            metrics.update(loss_fn_mapping)
            adversarial_loss_fn = MultiLosses([l for l in loss_fn_mapping.values()])

        # initialize discriminator
        self.adversarial_optimizer = adversarial_optimizer
        self.adversarial_loss_fn = adversarial_loss_fn

        # initialize super manager
        super().__init__(model, optimizer, loss_fn, metrics)


    def backward_adversarial(self, loss: torch.Tensor) -> None:
        """Backward pass for adversarial training"""
        assert self.adversarial_optimizer is not None, "adversarial optimizer is not compiled."
        loss.backward()
        clip_grad.clip_grad_norm_(self.model.parameters(), max_norm=1)

    def data_parallel(self, target_devices: list[torch.device]) -> bool:
        # wrap adversarial loss
        if self.adversarial_loss_fn is not None:
            self.adversarial_loss_fn, use_multi_gpus = devices.data_parallel(self.adversarial_loss_fn, target_devices, parallel_type=ParallelLoss)
        return use_multi_gpus and super().data_parallel(target_devices)

    def forward(self, input: DiffusionData, target: torch.Tensor | None = None) -> tuple[RegBBrgOutput | torch.Tensor, torch.Tensor | None]:
        # forward model
        y = cast(RegBBrgOutput, self.model(input))
        target_false = torch.ones_like(cast(torch.Tensor, y["d_false"]))

        # wrap output
        y = cast(torch.Tensor, y["target"]) if self.return_prediction else y

        # calculate loss
        target_dict = RegBBrgOutput(target=target, rec=input.condition, d_false=target_false, d_true=None)
        loss = self.compiled_losses(y, target_dict) if self.loss_fn is not None else None
        return y, loss

    def forward_discriminator(self, input: DiffusionData, _: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for discriminator"""
        # get true and false data
        y = cast(RegBBrgOutput, self.model(input, input.condition))
        y_false, y_true = y["d_false"], y["d_true"]
        assert y_false is not None and y_true is not None, "Discriminator output must be defined for adversarial learning."

        # adversarial learning labels
        target_false = torch.zeros_like(y_false)
        target_true = torch.ones_like(y_true)

        # concat data
        target_true = torch.cat([target_false, target_true], dim=0)
        y_true = torch.cat([y_false, y_true], dim=0)

        # calculate adversarial loss
        adv_loss = None if self.adversarial_loss_fn is None else self.compiled_adv_loss_fn(y_true, target_true)
        return y_true, adv_loss

    def reset(self, cpu: torch.device = devices.CPU) -> None:
        self.adversarial_loss_fn = self.raw_adv_loss_fn.to(cpu) if self.raw_adv_loss_fn is not None else self.raw_adv_loss_fn
        return super().reset(cpu)

    def test_step(self, x_test: torch.Tensor, y_test: torch.Tensor, *, forward_diffusion: bool = True) -> dict[str, float]:
        # forward diffusion
        if forward_diffusion:
            t = torch.full((x_test.shape[0],), self.time_steps, device=x_test.device)
            xt, objective = self.forward_diffusion(y_test, x_test, t)
            forward_diffusion = False
        else:
            xt, objective = x_test, y_test

        # model testing
        objective = y_test if self.return_prediction else objective
        return super().test_step(xt, objective, forward_diffusion=forward_diffusion)

    def to(self, device: torch.device) -> None:
        self.adversarial_loss_fn = None if self.adversarial_loss_fn is None else self.adversarial_loss_fn.to(device)
        return super().to(device)

    @overload
    def train_step(self, x_train: torch.Tensor, y_train: torch.Tensor) -> dict[str, float]:
        ...

    @overload
    def train_step(self, x_train: DiffusionData, y_train: torch.Tensor, *, forward_diffusion: bool = False) -> dict[str, float]:
        ...

    def train_step(self, x_train: torch.Tensor | DiffusionData, y_train: torch.Tensor, *, forward_diffusion: bool = True) -> dict[str, float]:
        # forward diffusion
        if forward_diffusion:
            assert isinstance(x_train, torch.Tensor), "x_train must be torch.Tensor for forward diffusion."
            xt, objective = self.forward_diffusion(y_train.to(x_train.device), x_train)
            xt = cast(DiffusionData, xt)
            objective = cast(torch.Tensor, objective)
        else:
            assert not isinstance(x_train, torch.Tensor), "x_train must be DiffusionData for backward diffusion."
            xt, objective = x_train, y_train

        # adversarial training
        if self.enable_adversarial_learning:
            _, adv_loss = self.forward_discriminator(xt, objective)
            self.compiled_adv_optimizer.zero_grad()
            assert adv_loss is not None, "Adversarial loss must be defined for adversarial training."
            self.backward_adversarial(adv_loss)
            self.compiled_adv_optimizer.step()

        # train generators
        return super().train_step(xt, objective, forward_diffusion=False)

    def use_prediction(self, return_prediction: bool) -> PredictionContext[M]:
        return PredictionContext(self.raw_model, return_prediction)

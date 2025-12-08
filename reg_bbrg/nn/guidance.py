import torch
from diffusion.data import DiffusionData
from torch.nn.parameter import Parameter
from typing import Any, Generic, Iterator, TypeVar, cast

from .bridge import ConditionalBrownianBridgeModule
from .protocols import RegBBrgOutput

Module = TypeVar('Module', bound=torch.nn.Module)
G = TypeVar('G', bound=torch.nn.Module)
D = TypeVar('D', bound=torch.nn.Module)


class RegularizedBrownianBridgeModule(ConditionalBrownianBridgeModule[Module, None, None], Generic[Module, G, D]):
    """
    The High-fidelity conditional Brownian Bridge.

    - Properties:
        - model: The diffusion model `Module` for generation mapping.
        - generator: The generator `G` for reconstruction mapping.
        - discriminator: The discriminator `DIS` for adversarial learning.
        - return_prediction: A `bool` flag of whether to return the prediction.
        - with_gen_time_emb: A `bool` flag of whether to use time embedding in the generator.
    """
    generator: G
    discriminator: D
    return_prediction: bool
    with_gen_time_emb: bool

    def __init__(self, diff_model: Module, time_steps: int, generator: G, discriminator: D, *, c_lambda: float = 2, with_gen_time_emb: bool = False) -> None:
        """
        Construct a PosteriorGuidanceABridgeModule.

        - Parameters:
            - diff_model: The UNET prediction `Module`.
            - time_steps: The number of time steps in `int` of the diffusion steps.
            - generator: The generator `G` for reconstruction mapping.
            - discriminator: The discriminator `DIS` for adversarial learning.
            - c_lambda: The lambda value for the ABridge in `float`.
            - encoder: The encoder `Module` for the ABridge.
            - decoder: The decoder `Module` for the ABridge.
            - with_gen_time_emb: A `bool` flag of whether to use time embedding in the generator.
        """
        super().__init__(diff_model, time_steps, c_lambda=c_lambda)
        self.generator = generator
        self.discriminator = discriminator
        self.return_prediction = False
        self.with_gen_time_emb = with_gen_time_emb

    def __call__(self, data: DiffusionData, real_data: torch.Tensor | None = None) -> RegBBrgOutput:
        return super().__call__(data, real_data)

    def forward(self, data: DiffusionData, real_data: torch.Tensor | None = None) -> RegBBrgOutput:
        # call the parent forward to get epsilon
        assert data.condition is not None, 'Condition data is required for Posterior Guidance A-Bridge.'
        epsilon = super().forward(data)
        x0_hat = data.x - epsilon

        # get xT_tilde from the generator
        xT_tilde = cast(torch.Tensor, self.generator(x0_hat, torch.zeros_like(data.t)) if self.with_gen_time_emb else self.generator(x0_hat))

        # direct return x0_hat if return prediction is enabled
        y = x0_hat if self.return_prediction else epsilon

        # check if real data is given
        if real_data is not None:
            xT_tilde = xT_tilde.detach()  # detach the generator output
            d_true = cast(torch.Tensor, self.discriminator(real_data))  # forward real data to discriminator
        else:
            d_true = None

        # forward fake data to discriminator
        d_false = cast(torch.Tensor, self.discriminator(xT_tilde))

        # zip data
        return RegBBrgOutput(target=y, rec=xT_tilde, d_false=d_false, d_true=d_true)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        params = list(self.model.parameters(recurse=recurse))
        if self.generator is not None:
            params += list(self.generator.parameters(recurse=recurse))
        return iter(params)

    def discriminator_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.discriminator.parameters(recurse=recurse)

    def sampling_step(self, data: DiffusionData, i: int, /, *, return_noise: bool = False, predicted_obj: RegBBrgOutput | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # unzip the predicted_obj to get target as epsilon
        epsilon = None if predicted_obj is None else predicted_obj['target']
        return super().sampling_step(data, i, return_noise=return_noise, predicted_obj=epsilon)

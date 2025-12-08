import torch
from typing import TypedDict


class RegBBrgOutput(TypedDict):
    target: torch.Tensor | None
    """The predicted target tensor of the diffusion model, epsilon<sub>θ</sub> when `return_prediction` is `False`, or x&#x302;<sub>0</sub> when `return_prediction` is `True`."""
    rec: torch.Tensor | None
    """The reconstructed tensor from the generator, x&#x303;<sub>T</sub>."""
    d_false: torch.Tensor | None
    """The discriminator output of the real data."""
    d_true: torch.Tensor | None
    """The discriminator output of the generated data."""

import torch
from diffusion.data import DiffusionData
from diffusion.nn.diffusion.protocols import TimedData
from typing import TypeVar

from sde_bbdm.nn import ABridgeModule as BrownianBridgeModule

Module = TypeVar('Module', bound=torch.nn.Module)
E = TypeVar('E', bound=torch.nn.Module | None)
D = TypeVar('D', bound=torch.nn.Module | None)


class ConditionalBrownianBridgeModule(BrownianBridgeModule[Module, E, D]):
    """A conditional Brownian Bridge module that takes in additional condition data."""
    def forward(self, data: TimedData) -> torch.Tensor:
        # unzip data
        x, t, condition = data.x, data.t, data.condition
        assert condition is not None, 'Condition data is required for Conditional Brownian Bridge.'

        # concat condition
        x = torch.cat([x, condition], dim=1)
        xt = DiffusionData(x, t)
        return super().forward(xt)

    def forward_diffusion(self, data: torch.Tensor, t: torch.Tensor | None = None, /, condition: torch.Tensor | None = None) -> tuple[DiffusionData, torch.Tensor]:
        # if condition shape is different from data, average the condition on channel dimension then expand
        assert condition is not None, 'Condition data is required for Conditional Brownian Bridge.'
        if condition.shape[1] != data.shape[1]:
            c = condition.mean(dim=1, keepdim=True).expand_as(data)
        else:
            c = condition

        # forward diffusion
        xt, obj = super().forward_diffusion(data, t, c)
        return DiffusionData(xt.x, xt.t, condition=condition), obj

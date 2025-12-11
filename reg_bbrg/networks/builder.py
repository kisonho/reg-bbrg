from diffusion.networks import UNet, build as build_unet
from sde_bbdm.networks import OpenAIUNet
from typing import overload

from .protocols import DiffusionModule, PixelDiscriminator
from .models import UNetType


@overload
def build(input_channels: int, output_channels: int, /, time_steps: int = 1000, *, model_type: UNetType = UNetType.A_BRIDGE) -> DiffusionModule[OpenAIUNet, UNet, PixelDiscriminator]: ...

@overload
def build(input_channels: int, output_channels: int, /, time_steps: int = 1000, *, model_type: UNetType = UNetType.A_BRIDGE, with_gen_time_emb: bool = True) -> DiffusionModule[OpenAIUNet, OpenAIUNet, PixelDiscriminator]: ...

def build(input_channels: int, output_channels: int, /, time_steps: int = 1000, *, model_type: UNetType = UNetType.A_BRIDGE, with_gen_time_emb: bool = False) -> DiffusionModule[OpenAIUNet, UNet | OpenAIUNet, PixelDiscriminator]:  # type: ignore
    """
    Build the A-Bridge model with cycle conditional GAN guidance.

    - Parameters:
        - input_channels: The number of input channels in `int`
        - output_channels: The number of output channels in `int`
        - time_steps: The number of time steps in `int`
        - c_lambda: The lambda value for A-Bridge model in `float`
        - model_type: The UNet model type in `UNetType`
    - Returns: The A-Bridge model with cycle conditional GAN guidance in `DiffusionModule` with main model as `OpenAIUNet`, generator as `UNet`, and discriminator as `PixelDiscriminator`.
    """
    # load model
    attention_resolutions, channel_mults = model_type.value
    unet = OpenAIUNet(input_channels + output_channels, 128, output_channels, num_res_blocks=2, attention_resolutions=attention_resolutions, channel_mult=channel_mults, num_heads=8, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True)
    generator = OpenAIUNet(output_channels, 128, input_channels, num_res_blocks=2, attention_resolutions=attention_resolutions, channel_mult=channel_mults, num_heads=8, num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True) if with_gen_time_emb else build_unet(output_channels, input_channels, dim_mults=channel_mults, with_time_emb=False)
    discriminator = PixelDiscriminator(input_channels)
    return DiffusionModule(unet, time_steps, generator=generator, discriminator=discriminator, with_gen_time_emb=with_gen_time_emb)

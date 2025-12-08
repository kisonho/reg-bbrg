import torch
from typing import Type


class PixelDiscriminator(torch.nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
    net: torch.nn.Sequential

    def __init__(self, input_nc: int, ndf: int = 64, norm_layer: Type[torch.nn.Module] = torch.nn.BatchNorm2d, use_bias: bool = False):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            use_bias (bool) -- if the conv layer uses bias or not
        """
        super(PixelDiscriminator, self).__init__()
        net = [
            torch.nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            torch.nn.LeakyReLU(0.2, True),
            torch.nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = torch.nn.Sequential(*net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


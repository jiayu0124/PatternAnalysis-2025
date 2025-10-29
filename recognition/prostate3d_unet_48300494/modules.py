"""
modules.py
-----------

This file contains the implementation of a simple 3‑D U‑Net architecture
for volumetric segmentation tasks.  The network is composed of a series
of down‑sampling blocks followed by corresponding up‑sampling blocks
with skip connections.  A small number of base channels and a modest
number of levels are used to keep the model lightweight enough for
practice on a single GPU or even a CPU.

The architecture can be instantiated with different numbers of input
channels and target classes.  For normal difficulty as specified in
COMP3710, a two‑layer encoder–decoder with a base width of 32 is
adequate to achieve a Dice score above 0.7 on the prostate dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two consecutive 3×3×3 convolutions with batch normalisation and ReLU.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.

    Returns
    -------
    nn.Sequential
        The convolutional block.
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    """A down‑sampling block consisting of max pooling followed by a conv block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = _conv_block(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """An up‑sampling block using transposed convolution followed by a conv block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Use ConvTranspose3d to upsample by a factor of 2
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = _conv_block(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad the upsampled tensor if necessary to match skip connection size
        diff_z = skip.size(2) - x.size(2)
        diff_y = skip.size(3) - x.size(3)
        diff_x = skip.size(4) - x.size(4)
        x = F.pad(x, (0, diff_x, 0, diff_y, 0, diff_z))
        # Concatenate along channel dimension
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """A 3‑D U‑Net for volumetric segmentation.

    Parameters
    ----------
    in_channels : int, optional
        Number of channels in the input volume.  Defaults to 1.
    num_classes : int, optional
        Number of output segmentation classes.  Defaults to 5.
    base_channels : int, optional
        Number of feature channels in the first layer.  Defaults to 32.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 5, base_channels: int = 32) -> None:
        super().__init__()
        # Encoder
        self.inc = _conv_block(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        # Final 1×1×1 convolution to get per‑voxel class logits
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
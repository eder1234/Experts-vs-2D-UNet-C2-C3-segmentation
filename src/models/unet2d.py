# src/models/unet2d.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------#
class DoubleConv(nn.Sequential):
    """(conv → BN → ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)


class Down(nn.Sequential):
    """Down-scaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))


class Up(nn.Module):
    """
    Up-scaling then DoubleConv.

    When `bilinear=True`, uses interpolation + 1×1 conv for channel align.
    Otherwise uses ConvTranspose2d.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            )
            conv_in = in_channels // 2 + out_channels
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            conv_in = in_channels // 2 + out_channels

        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x1 to the same size as x2 (in case of odd dims)
        diff_y = x2.size(-2) - x1.size(-2)
        diff_x = x2.size(-1) - x1.size(-1)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Conv2d):
    """1×1 convolution to get the desired number of output channels"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels, kernel_size=1)


# ---------------------------------------------------------------------------#
class UNet2D(nn.Module):
    """A lightweight 4-level U-Net for 2-D segmentation."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        base_channels: int = 32,
        bilinear: bool = False,
    ):
        super().__init__()

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        self.outc = OutConv(base_channels, out_channels)

    # --------------------------------------------------------------------- #
    def forward(self, x):
        x1 = self.inc(x)     # 1/1
        x2 = self.down1(x1)  # 1/2
        x3 = self.down2(x2)  # 1/4
        x4 = self.down3(x3)  # 1/8
        x5 = self.down4(x4)  # 1/16
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # no activation here
        return logits

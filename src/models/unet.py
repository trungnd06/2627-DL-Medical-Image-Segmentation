"""M1: U-Net tự cài, khởi tạo ngẫu nhiên, không dùng pretrained weights.

Encoder mặc định: 32 -> 64 -> 128 -> 256, bottleneck: 512.
Đầu vào [B, 3, H, W]; đầu ra logits [B, 1, H, W], chưa sigmoid.
"""

from numbers import Integral

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Hai convolution 3x3 giữ kích thước, mỗi lớp theo sau bởi ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """Upsample bằng transposed convolution, nối skip rồi DoubleConv."""

    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(2 * skip_channels, skip_channels)

    def forward(self, x, skip):
        # output_size chọn output_padding phù hợp khi kích thước ảnh là số lẻ.
        x = self.up(x, output_size=skip.shape)
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """U-Net với bốn mức downsampling, dùng cho baseline M1.

    channels gồm bốn mức encoder và một bottleneck, khớp configs/unet.yaml.
    Không BatchNorm, dropout hoặc activation cuối. Loss BCEWithLogits nhận
    trực tiếp logits; sigmoid/threshold chỉ dùng khi đánh giá.
    H, W tối thiểu 16; kích thước đầu ra luôn bằng đầu vào, kể cả số lẻ.
    Model không tự resize, normalize hoặc loại padding; Dataset và loss
    đảm nhiệm các phần đó bằng geometry/valid_mask.
    """

    def __init__(self, in_channels=3, output_channels=1, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        channels = tuple(channels)
        if len(channels) != 5:
            raise ValueError("channels must contain four encoder widths and one bottleneck width")
        if any(isinstance(c, bool) or not isinstance(c, Integral) or c <= 0
               for c in (in_channels, output_channels, *channels)):
            raise ValueError("Channel counts must be positive integers")
        self.in_channels = in_channels
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels[:-1]:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels[-2], channels[-1])
        self.decoders = nn.ModuleList(
            UpBlock(channels[index + 1], channels[index])
            for index in range(3, -1, -1)
        )
        self.head = nn.Conv2d(channels[0], output_channels, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B, {self.in_channels}, H, W] input")
        if min(x.shape[-2:]) < 16:
            raise ValueError("Input height and width must be at least 16 for four pooling levels")
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.head(x)

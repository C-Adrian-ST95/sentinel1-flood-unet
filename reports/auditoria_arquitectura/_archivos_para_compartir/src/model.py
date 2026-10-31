"""U-Net original: 2 canales SAR (VV/VH) -> 1 logit de agua por píxel."""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Convolución 3x3 -> BatchNorm -> ReLU (dos veces)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    """Reduce a la mitad la resolución espacial."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Interpola, concatena skip connection y procesa las características."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    """Arquitectura encoder-decoder de 3 reducciones para prueba de entrenamiento."""

    def __init__(self, in_channels=2, out_channels=1, base=16):
        super().__init__()
        self.encoder0 = DoubleConv(in_channels, base)
        self.encoder1 = Down(base, base * 2)
        self.encoder2 = Down(base * 2, base * 4)
        self.encoder3 = Down(base * 4, base * 8)
        self.decoder1 = Up(base * 8, base * 4, base * 4)
        self.decoder2 = Up(base * 4, base * 2, base * 2)
        self.decoder3 = Up(base * 2, base, base)
        self.output = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        a = self.encoder0(x)
        b = self.encoder1(a)
        c = self.encoder2(b)
        d = self.encoder3(c)
        x = self.decoder1(d, c)
        x = self.decoder2(x, b)
        x = self.decoder3(x, a)
        return self.output(x)  # Sin sigmoid; BCEWithLogitsLoss lo incorpora.

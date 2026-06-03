import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseSegmentationModel
from models.unet import DoubleConv

class AttentionGate(nn.Module):
    """
    Attention Gate for Attention U-Net.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        # Gating signal path
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        # Skip connection path
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g: gating signal, x: skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample g to match x size if necessary
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttentionUNet(BaseSegmentationModel):
    """
    Manual implementation of the Attention U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(AttentionUNet, self).__init__(in_channels, out_channels)

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Attention Gates
        self.ag1 = AttentionGate(F_g=1024, F_l=512, F_int=256)
        self.ag2 = AttentionGate(F_g=512, F_l=256, F_int=128)
        self.ag3 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.ag4 = AttentionGate(F_g=128, F_l=64, F_int=32)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Attention
        g = x5
        x = self.up1(g)
        x_att = self.ag1(g, x4)
        x = torch.cat([x, x_att], dim=1)
        x = self.conv_up1(x)

        g = x
        x = self.up2(g)
        x_att = self.ag2(g, x3)
        x = torch.cat([x, x_att], dim=1)
        x = self.conv_up2(x)

        g = x
        x = self.up3(g)
        x_att = self.ag3(g, x2)
        x = torch.cat([x, x_att], dim=1)
        x = self.conv_up3(x)

        g = x
        x = self.up4(g)
        x_att = self.ag4(g, x1)
        x = torch.cat([x, x_att], dim=1)
        x = self.conv_up4(x)

        return self.outc(x)

import torch
import torch.nn as nn

class CAM_ACR(nn.Module):
    """
    Channel Attention Module with Adaptive Channel Reordering
    """
    def __init__(self, channels, reduction=16, groups=32):
        super().__init__()
        self.channels = channels
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_stack = nn.Sequential(
            nn.Linear(channels, max(1, channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channels // reduction), channels * 2),
            nn.Sigmoid()
        )
        self.grouped_conv = nn.Conv2d(channels, channels, kernel_size=3,
                                      padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled_x = self.gap(x).view(b, c)
        alpha_beta = self.fc_stack(pooled_x).view(b, c * 2, 1, 1)
        alpha = alpha_beta[:, :c, :, :]
        beta = alpha_beta[:, c:, :, :]
        x_att = alpha * x + beta
        x_local = self.conv1x1(self.gelu(self.norm(self.grouped_conv(x_att))))
        return x_local

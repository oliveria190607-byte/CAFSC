import torch
import torch.nn as nn

class LocalGlobalFeatureEnhancement(nn.Module):
    """
    Local-Global Feature Enhancement Module
    """
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, f_global, f_local):
        f_interaction = f_global * f_local
        r_prime = torch.sigmoid(f_interaction)
        f_s = self.gap(f_global)
        att_input = f_s.expand_as(r_prime) * r_prime
        attention_map = self.conv1x1(att_input)
        f_guided = attention_map * f_global + f_global
        return f_guided

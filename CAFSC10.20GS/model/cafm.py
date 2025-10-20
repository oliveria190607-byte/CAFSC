import torch
import torch.nn as nn

class CAFM(nn.Module):
    """
    Cross Attention Fusion Module
    """
    def __init__(self, channels):
        super().__init__()
        self.conv_global_coeff = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_local_coeff = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, f_guided, f_local):
        r_cross_coeff_input = f_guided * f_local
        r_cross_coeff = torch.sigmoid(r_cross_coeff_input)
        f_interaction_local = r_cross_coeff * f_local
        w_global_coeff = torch.sigmoid(self.conv_global_coeff(f_guided))
        w_local_coeff = torch.sigmoid(self.conv_local_coeff(f_interaction_local))
        fused_global_component = w_global_coeff * f_guided
        fused_local_component = w_local_coeff * f_local
        fused_features = fused_global_component + fused_local_component
        return fused_features

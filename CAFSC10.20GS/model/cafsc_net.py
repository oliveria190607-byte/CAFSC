import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from model.cam_acr import CAM_ACR
from model.local_global_enhancement import LocalGlobalFeatureEnhancement
from model.cafm import CAFM


class CAFSC_Net(nn.Module):
    """
    CAFSC Network: Full model integrating CAM-ACR, Local-Global Enhancement, and CAFM
    """

    def __init__(self, num_classes=7, resnet_feature_layer='layer3', pretrained_resnet=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_resnet else None
        base = resnet50(weights=weights)

        # 特征提取器
        if resnet_feature_layer == 'layer3':
            self.feature_extractor = nn.Sequential(*list(base.children())[:-3])
            feature_channels = 1024
        else:  # 默认 layer4
            self.feature_extractor = nn.Sequential(*list(base.children())[:-2])
            feature_channels = 2048

        # 模块
        self.cam_acr = CAM_ACR(channels=feature_channels)
        self.local_global_enhancement = LocalGlobalFeatureEnhancement(channels=feature_channels)
        self.cafm = CAFM(channels=feature_channels)

        # 分类头
        self.final_gap = nn.AdaptiveAvgPool2d(1)
        self.final_fc = nn.Linear(feature_channels, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        f_resnet = self.feature_extractor(x)
        f_local = self.cam_acr(f_resnet)
        f_guided = self.local_global_enhancement(f_resnet, f_local)
        fused = self.cafm(f_guided, f_local)
        out = self.final_gap(fused)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.final_fc(out)
        return out

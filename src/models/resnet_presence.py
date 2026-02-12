#!/usr/bin/env python3
from typing import Optional

import torch.nn as nn
from torchvision import models


def build_resnet_presence(
    backbone: str = 'resnet18',
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    backbone = backbone.lower()

    weights: Optional[object] = None
    if pretrained:
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        elif backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

    if backbone == 'resnet18':
        model = models.resnet18(weights=weights)
    elif backbone == 'resnet34':
        model = models.resnet34(weights=weights)
    elif backbone == 'resnet50':
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f'Unsupported backbone: {backbone}')

    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, 1))
    else:
        model.fc = nn.Linear(in_features, 1)

    return model

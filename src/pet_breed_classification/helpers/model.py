"""
Device selection and model construction.
"""

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)

from .. import config


def get_device() -> torch.device:
    """CUDA if available (e.g. Colab), otherwise CPU (e.g. local machine) —
    same code runs unchanged in both places.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(backbone_name: str) -> nn.Module:
    """Build an ImageNet-pretrained backbone with its final classifier
    layer replaced for config.NUM_CLASSES breeds. Explicit and simple:
    each weight set is named directly, and torchvision downloads it to
    its local cache automatically the first time this runs.

    ResNet-family models expose their final layer as `.fc`; MobileNetV3
    exposes it as the last layer of its `.classifier` Sequential — that's
    the one real structural difference handled below.
    """
    if backbone_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model

    if backbone_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
        return model

    if backbone_name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, config.NUM_CLASSES)
        return model

    raise ValueError(
        f"Unknown backbone '{backbone_name}'. Supported: {config.SUPPORTED_BACKBONES}"
    )
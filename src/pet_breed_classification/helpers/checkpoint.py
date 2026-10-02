
import json

import torch
from torch import nn

from .. import config


def save_checkpoint(model: nn.Module, backbone_name: str) -> tuple[str, str]:
    """Save model weights plus a sidecar JSON describing exactly what
    preprocessing the checkpoint expects — the concrete record of "export
    the eval transform alongside the weights".
    """
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = config.CHECKPOINT_DIR / f"{backbone_name}_best.pt"
    transform_path = config.CHECKPOINT_DIR / f"{backbone_name}_best_transform.json"

    torch.save(model.state_dict(), weights_path)

    transform_metadata = {
        "resize_size": config.RESIZE_SIZE,
        "image_size": config.IMAGE_SIZE,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
        "num_classes": config.NUM_CLASSES,
    }
    with transform_path.open("w") as f:
        json.dump(transform_metadata, f, indent=2)

    return str(weights_path), str(transform_path)


def load_checkpoint(model: nn.Module, backbone_name: str, device: torch.device) -> dict:
    """Load saved weights into `model` (in place) and return the transform
    metadata dict that was saved alongside them, so callers can confirm
    what preprocessing the checkpoint expects.
    """
    weights_path = config.CHECKPOINT_DIR / f"{backbone_name}_best.pt"
    transform_path = config.CHECKPOINT_DIR / f"{backbone_name}_best_transform.json"

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with transform_path.open() as f:
        transform_metadata = json.load(f)

    return transform_metadata
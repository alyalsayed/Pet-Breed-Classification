"""
Saving and loading trained model checkpoints.
"""

import json

import torch
from torch import nn

from .. import config


def save_checkpoint(
    model: nn.Module, backbone_name: str, checkpoint_dir=None
) -> tuple[str, str]:
    """Save model weights plus a sidecar JSON describing exactly what
    preprocessing the checkpoint expects — the concrete record of "export
    the eval transform alongside the weights".

    checkpoint_dir defaults to config.CHECKPOINT_DIR; pass a different
    path to save into an isolated location (e.g. a specific candidate
    run's folder) without touching the shared default location.
    """
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_path = checkpoint_dir / f"{backbone_name}_best.pt"
    transform_path = checkpoint_dir / f"{backbone_name}_best_transform.json"

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


def load_checkpoint(
    model: nn.Module, backbone_name: str, device: torch.device, checkpoint_dir=None
) -> dict:
    """Load saved weights into `model` (in place) and return the transform
    metadata dict that was saved alongside them, so callers can confirm
    what preprocessing the checkpoint expects.

    checkpoint_dir defaults to config.CHECKPOINT_DIR; pass a different
    path to load a specific candidate checkpoint (e.g. one downloaded
    from a particular MLflow run) instead of the shared default.
    """
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    weights_path = checkpoint_dir / f"{backbone_name}_best.pt"
    transform_path = checkpoint_dir / f"{backbone_name}_best_transform.json"

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with transform_path.open() as f:
        transform_metadata = json.load(f)

    return transform_metadata
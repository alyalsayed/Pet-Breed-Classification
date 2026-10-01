import json
import torch
import torch.nn as nn
import pet_breed_classification.config as config
# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

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


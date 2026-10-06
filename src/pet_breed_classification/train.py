"""
Baseline training: ResNet-50 on the Pet Breed manifest, logged to MLflow.

Loads train/val records from the manifest (never test), fine-tunes an
ImageNet-pretrained ResNet-50, and tracks the best validation checkpoint.
Preprocessing reuses features.py's shared constants/loader so training
and future serving never drift apart.

Run:
    uv run train                                  # defaults to resnet50
    uv run train --backbone resnet18
    uv run train --backbone mobilenet_v3_small
    # or directly:
    uv run python -m pet_breed_classification.train --backbone resnet18
"""

import argparse

import mlflow
import mlflow.pytorch
import torch
from torch import nn

from . import config
from .helpers.checkpoint import load_checkpoint, save_checkpoint
from .helpers.dataset import build_dataloaders
from .helpers.engine import train_one_epoch, validate
from .helpers.model import build_model, get_device

MODEL_VERSION = "v1"
DATASET_NAME = "oxford-iiit-pet"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=config.SUPPORTED_BACKBONES,
        default="resnet50",
        help="Which backbone to train (default: resnet50).",
    )
    args = parser.parse_args()
    backbone_name = args.backbone

    device = get_device()
    print(f"Using device: {device}")
    print(f"Backbone: {backbone_name}")

    train_loader, val_loader = build_dataloaders()
    model = build_model(backbone_name).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_top1 = 0.0

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        # Tags make runs easy to find/filter in the MLflow UI later
        # (e.g. "show me every v1 run trained on oxford-iiit-pet").
        mlflow.set_tags(
            {
                "model_version": MODEL_VERSION,
                "dataset": DATASET_NAME,
            }
        )

        mlflow.log_params(
            {
                "backbone": backbone_name,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "num_epochs": config.NUM_EPOCHS,
                "split_seed": config.SPLIT_SEED,
                "num_classes": config.NUM_CLASSES,
            }
        )

        for epoch in range(1, config.NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = validate(model, val_loader, criterion, device)

            print(
                f"epoch {epoch}/{config.NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_top1={val_metrics['val_top1']:.4f} | "
                f"val_macro_f1={val_metrics['val_macro_f1']:.4f} | "
                f"val_ece={val_metrics['val_ece']:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for name, value in val_metrics.items():
                mlflow.log_metric(name, value, step=epoch)

            if val_metrics["val_top1"] > best_val_top1:
                best_val_top1 = val_metrics["val_top1"]

                
                weights_path, transform_path = save_checkpoint(model, backbone_name)
                mlflow.log_artifact(weights_path)
                mlflow.log_artifact(transform_path)

        mlflow.log_metric("best_val_top1", best_val_top1)
        
        load_checkpoint(model, backbone_name, device)

        example_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
        mlflow.pytorch.log_model(model, name="model", input_example=example_input)

        print(f"Done. Best val_top1={best_val_top1:.4f}")


if __name__ == "__main__":
    main()
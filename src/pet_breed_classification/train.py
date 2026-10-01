"""
Baseline training: ResNet-50 on the Pet Breed manifest, logged to MLflow.

Loads train/val records from the manifest (never test), fine-tunes an
ImageNet-pretrained ResNet-50, and tracks the best validation checkpoint.
Preprocessing reuses features.py's shared constants/loader so training
and future serving never drift apart.

Run:
    uv run train
    # or directly:
    uv run python -m pet_breed_classification.train
"""



import mlflow
import torch
from torch import nn
from . import config
from .helpers.dataset import build_dataloaders
from .helpers.model import build_model, get_device
from .helpers.engine import train_one_epoch, validate
from .helpers.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()
    model = build_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    backbone_name = "resnet50"
    best_val_top1 = 0.0

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "backbone": backbone_name,
                "learning_rate": config.LEARNING_RATE,
                "batch_size": config.BATCH_SIZE,
                "num_epochs": config.NUM_EPOCHS,
                "split_seed": config.SPLIT_SEED,
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
        print(f"Done. Best val_top1={best_val_top1:.4f}")


if __name__ == "__main__":
    main()
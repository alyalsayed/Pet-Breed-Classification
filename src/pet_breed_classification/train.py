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

import json

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50

from pet_breed_classification import config, features


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """CUDA if available (e.g. Colab), otherwise CPU (e.g. local WSL) —
    same code runs unchanged in both places.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PetBreedDataset(Dataset):
    """Wraps a list of manifest records (already filtered to one split)."""

    def __init__(self, records: list[dict], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = features.load_image(record["path"])
        image = self.transform(image)
        label = record["class_index"]
        return image, label


def load_split_records(split_name: str) -> list[dict]:
    with config.MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    return [record for record in manifest if record["split"] == split_name]


def build_train_transform() -> transforms.Compose:
    """Same resize/crop/normalize as features.get_eval_transform(), with one
    light augmentation (random horizontal flip) added on top. Augmentation
    stays here, in training code only — never in features.py, and never
    applied to validation/test/serving.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_records = load_split_records("train")
    val_records = load_split_records("val")

    train_ds = PetBreedDataset(train_records, build_train_transform())
    val_ds = PetBreedDataset(val_records, features.get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """ResNet-50 with ImageNet-pretrained weights, explicit and simple:
    the exact weight set is named directly, and torchvision downloads it
    to its local cache automatically the first time this runs — no extra
    download logic needed.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    return model


# ---------------------------------------------------------------------------
# Metrics (plain Python/torch — no extra dependency)
# ---------------------------------------------------------------------------

def compute_top1_accuracy(all_preds: list[int], all_labels: list[int]) -> float:
    correct = sum(p == y for p, y in zip(all_preds, all_labels))
    return correct / len(all_labels)


def compute_macro_f1(all_preds: list[int], all_labels: list[int], num_classes: int) -> float:
    """Macro-F1: compute F1 per class, then average — every breed counts
    equally regardless of how many validation images it has.
    """
    f1_scores = []
    for class_idx in range(num_classes):
        tp = sum(p == class_idx and y == class_idx for p, y in zip(all_preds, all_labels))
        fp = sum(p == class_idx and y != class_idx for p, y in zip(all_preds, all_labels))
        fn = sum(p != class_idx and y == class_idx for p, y in zip(all_preds, all_labels))

        if tp == 0 and (fp == 0 or fn == 0):
            # No predictions and/or no true examples for this class this epoch.
            f1_scores.append(0.0)
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


def compute_ece(all_confidences: list[float], all_correct: list[bool], num_bins: int) -> float:
    """Expected Calibration Error, using max-softmax confidence as the
    predicted probability. This is the UNCALIBRATED baseline number —
    temperature scaling (a later step) is what actually improves it.
    """
    bin_edges = [i / num_bins for i in range(num_bins + 1)]
    n = len(all_confidences)
    ece = 0.0

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            (conf, correct)
            for conf, correct in zip(all_confidences, all_correct)
            if (lo < conf <= hi) or (i == 0 and conf == lo)
        ]
        if not in_bin:
            continue
        bin_confidence = sum(c for c, _ in in_bin) / len(in_bin)
        bin_accuracy = sum(correct for _, correct in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_accuracy - bin_confidence)

    return ece


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_confidences, all_correct = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_correct.extend((preds == labels).cpu().tolist())

    return {
        "val_loss": total_loss / len(loader.dataset),
        "val_top1": compute_top1_accuracy(all_preds, all_labels),
        "val_macro_f1": compute_macro_f1(all_preds, all_labels, config.NUM_CLASSES),
        "val_ece": compute_ece(all_confidences, all_correct, config.ECE_NUM_BINS),
    }


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
"""
Calibrate the trained model using temperature scaling, on the VALIDATION
set only — never touches test, never retrains the model.

Computes ECE before and after calibration, saves a reliability diagram,
fits an abstention threshold, builds a confusion matrix, and saves all
results to disk.

Run:
    uv run calibrate
    uv run calibrate --checkpoint-dir models/candidate_masked-gnu-592
    # or directly:
    uv run python -m pet_breed_classification.calibrate
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from . import config
from .helpers.calibration import (
    collect_logits_and_labels,
    confidences_and_correctness,
    ece_at_temperature,
    fit_temperature,
    reliability_diagram_data,
)
from .helpers.checkpoint import load_checkpoint
from .helpers.confusion import build_confusion_matrix, most_confused_pairs
from .helpers.dataset import build_val_loader
from .helpers.model import build_model, get_device
from .helpers.selective import compute_selective_metrics, find_threshold_for_target_accuracy
from .manifest import load_index_to_breed

BACKBONE_NAME = "resnet50"


def plot_reliability_diagram(
    before: tuple[list[float], list[float]],
    after: tuple[list[float], list[float]],
    output_path: Path,
) -> None:
    """Saves a two-panel before/after reliability diagram. A perfectly
    calibrated model's points would lie exactly on the diagonal.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, (confidences, accuracies), title in zip(
        axes, [before, after], ["Before calibration", "After calibration"]
    ):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        ax.plot(confidences, accuracies, marker="o", color="tab:blue", label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: list[list[int]], breed_names: list[str], output_path: Path
) -> None:
    """Saves the full 37x37 confusion matrix as a heatmap. Breed names are
    small on a 37-class axis, so this is meant as a visual/report artifact
    (spotting bright off-diagonal cells), not for reading individual labels —
    most_confused_pairs() is the readable version of the same data.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(confusion_matrix, cmap="viridis")

    ax.set_xticks(range(len(breed_names)))
    ax.set_yticks(range(len(breed_names)))
    ax.set_xticklabels(breed_names, rotation=90, fontsize=6)
    ax.set_yticklabels(breed_names, fontsize=6)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title("Validation set confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <backbone>_best.pt / _transform.json. "
            "Defaults to config.CHECKPOINT_DIR. Pass a candidate-specific "
            "directory (e.g. models/candidate_masked-gnu-592) to calibrate "
            "that exact checkpoint instead of the shared default."
        ),
    )
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir or config.CHECKPOINT_DIR

    calibration_report_path = checkpoint_dir / config.CALIBRATION_REPORT_PATH.name
    confusion_matrix_report_path = checkpoint_dir / config.CONFUSION_MATRIX_REPORT_PATH.name
    calibration_results_path = checkpoint_dir / config.CALIBRATION_RESULTS_PATH.name

    device = get_device()
    print(f"Using device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    model = build_model(BACKBONE_NAME).to(device)
    load_checkpoint(model, BACKBONE_NAME, device, checkpoint_dir=checkpoint_dir)

    val_loader = build_val_loader()
    logits, labels = collect_logits_and_labels(model, val_loader, device)

    ece_before = ece_at_temperature(logits, labels, temperature=1.0, num_bins=config.ECE_NUM_BINS)
    print(f"ECE before calibration: {ece_before:.4f}")

    temperature = fit_temperature(logits, labels)
    print(f"Learned temperature: {temperature:.4f}")

    ece_after = ece_at_temperature(logits, labels, temperature=temperature, num_bins=config.ECE_NUM_BINS)
    print(f"ECE after calibration:  {ece_after:.4f}")

    before_diagram = reliability_diagram_data(logits, labels, temperature=1.0, num_bins=config.ECE_NUM_BINS)
    after_diagram = reliability_diagram_data(logits, labels, temperature=temperature, num_bins=config.ECE_NUM_BINS)
    plot_reliability_diagram(before_diagram, after_diagram, calibration_report_path)
    print(f"Saved reliability diagram to {calibration_report_path}")

    confidences, correct = confidences_and_correctness(logits, labels, temperature)
    threshold = find_threshold_for_target_accuracy(
        confidences, correct, target_accuracy=config.TARGET_SELECTIVE_ACCURACY
    )
    selective_metrics = compute_selective_metrics(confidences, correct, threshold)

    print(f"\nAbstention threshold (target selective accuracy={config.TARGET_SELECTIVE_ACCURACY}):")
    print(f"  threshold:           {selective_metrics['threshold']:.4f}")
    print(f"  coverage:            {selective_metrics['coverage']:.4f}")
    print(f"  selective_accuracy:  {selective_metrics['selective_accuracy']:.4f}")
    print(f"  overall_accuracy:    {selective_metrics['overall_accuracy']:.4f}")

    index_to_breed = load_index_to_breed()
    num_classes = len(index_to_breed)
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    true_labels = labels.tolist()

    confusion_matrix = build_confusion_matrix(true_labels, predicted_labels, num_classes)
    breed_names = [index_to_breed[i] for i in range(num_classes)]
    plot_confusion_matrix(confusion_matrix, breed_names, confusion_matrix_report_path)
    print(f"Saved confusion matrix to {confusion_matrix_report_path}")

    top_confused = most_confused_pairs(confusion_matrix, index_to_breed, config.TOP_CONFUSED_PAIRS_COUNT)
    print(f"\nTop {config.TOP_CONFUSED_PAIRS_COUNT} most confused breed pairs (true -> predicted: count):")
    for true_breed, predicted_breed, count in top_confused:
        print(f"  {true_breed} -> {predicted_breed}: {count}")

    results = {
        "backbone": BACKBONE_NAME,
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "target_selective_accuracy": config.TARGET_SELECTIVE_ACCURACY,
        **selective_metrics,
        "top_confused_pairs": [list(pair) for pair in top_confused],
    }
    calibration_results_path.parent.mkdir(parents=True, exist_ok=True)
    with calibration_results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {calibration_results_path}")


if __name__ == "__main__":
    main()
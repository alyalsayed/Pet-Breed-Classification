"""
Calibrate the trained model using temperature scaling, on the VALIDATION
set only — never touches test, never retrains the model.

Computes ECE before and after calibration, saves a reliability diagram,
fits an abstention threshold, builds a confusion matrix, and saves all
results to disk.

Run:
    uv run calibrate
    # or directly:
    uv run python -m pet_breed_classification.calibrate
"""

import json

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


def plot_reliability_diagram(before: tuple[list[float], list[float]], after: tuple[list[float], list[float]]) -> None:
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
    config.CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.CALIBRATION_REPORT_PATH)
    plt.close(fig)


def plot_confusion_matrix(confusion_matrix: list[list[int]], breed_names: list[str]) -> None:
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
    config.CONFUSION_MATRIX_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.CONFUSION_MATRIX_REPORT_PATH)
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    load_checkpoint(model, BACKBONE_NAME, device)

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
    plot_reliability_diagram(before_diagram, after_diagram)
    print(f"Saved reliability diagram to {config.CALIBRATION_REPORT_PATH}")

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

    # Confusion matrix — reuses the SAME logits/labels already computed
    # above. Temperature scaling never changes argmax, so no new model
    # inference is needed here.
    index_to_breed = load_index_to_breed()
    num_classes = len(index_to_breed)
    predicted_labels = torch.argmax(logits, dim=1).tolist()
    true_labels = labels.tolist()

    confusion_matrix = build_confusion_matrix(true_labels, predicted_labels, num_classes)
    breed_names = [index_to_breed[i] for i in range(num_classes)]
    plot_confusion_matrix(confusion_matrix, breed_names)
    print(f"Saved confusion matrix to {config.CONFUSION_MATRIX_REPORT_PATH}")

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
    config.CALIBRATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CALIBRATION_RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {config.CALIBRATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
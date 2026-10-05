"""
Calibrate the trained model using temperature scaling, on the VALIDATION
set only — never touches test, never retrains the model.

Computes ECE before and after calibration, saves a reliability diagram,
and saves the learned temperature + results to disk.

Run:
    uv run calibrate
    # or directly:
    uv run python -m pet_breed_classification.calibrate
"""

import json

import matplotlib.pyplot as plt

from . import config
from .helpers.calibration import (
    collect_logits_and_labels,
    ece_at_temperature,
    fit_temperature,
    reliability_diagram_data,
)
from .helpers.checkpoint import load_checkpoint
from .helpers.dataset import build_val_loader
from .helpers.model import build_model, get_device

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

    results = {
        "backbone": BACKBONE_NAME,
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    config.CALIBRATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CALIBRATION_RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved calibration results to {config.CALIBRATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
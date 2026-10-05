"""
Unit tests for helpers/calibration.py, using small synthetic logits/labels
(no dataset, no trained model needed) — fast and fully deterministic.

Run:
    uv run pytest tests/test_calibration.py
"""

import torch

from pet_breed_classification.helpers.calibration import (
    confidences_and_correctness,
    ece_at_temperature,
    fit_temperature,
    reliability_diagram_data,
)

NUM_BINS = 10


def make_overconfident_logits() -> tuple[torch.Tensor, torch.Tensor]:
    """3-class problem, 20 examples, where the model is right only ~60% of
    the time but its raw logits are extremely peaked (near-100% confidence
    every time) — a deliberately overconfident, miscalibrated setup.
    """
    torch.manual_seed(0)
    num_examples = 20
    num_classes = 3

    labels = torch.randint(0, num_classes, (num_examples,))
    logits = torch.zeros(num_examples, num_classes)

    for i, label in enumerate(labels):
        correct = torch.rand(1).item() < 0.6
        predicted_class = label.item() if correct else (label.item() + 1) % num_classes
        logits[i, predicted_class] = 10.0  # very peaked -> ~100% softmax confidence

    return logits, labels


def test_fit_temperature_reduces_ece_on_overconfident_model():
    logits, labels = make_overconfident_logits()

    ece_before = ece_at_temperature(logits, labels, temperature=1.0, num_bins=NUM_BINS)
    temperature = fit_temperature(logits, labels, num_steps=200)
    ece_after = ece_at_temperature(logits, labels, temperature=temperature, num_bins=NUM_BINS)

    assert temperature > 1.0, "an overconfident model should need T > 1 to soften confidence"
    assert ece_after < ece_before, "calibration should reduce ECE on a miscalibrated model"


def test_temperature_scaling_does_not_change_predictions():
    """Dividing all logits by a positive scalar never changes which class
    has the highest score — accuracy must be identical before and after.
    """
    logits, labels = make_overconfident_logits()

    _, correct_before = confidences_and_correctness(logits, labels, temperature=1.0)
    _, correct_after = confidences_and_correctness(logits, labels, temperature=2.5)

    assert correct_before == correct_after


def test_confidences_and_correctness_shapes():
    logits, labels = make_overconfident_logits()
    confidences, correct = confidences_and_correctness(logits, labels, temperature=1.0)

    assert len(confidences) == len(labels)
    assert len(correct) == len(labels)
    assert all(0.0 <= c <= 1.0 for c in confidences)


def test_reliability_diagram_data_stays_within_bounds():
    logits, labels = make_overconfident_logits()
    confidences, accuracies = reliability_diagram_data(logits, labels, temperature=1.0, num_bins=NUM_BINS)

    assert len(confidences) == len(accuracies)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert all(0.0 <= a <= 1.0 for a in accuracies)
"""
Unit tests for helpers/selective.py, using small synthetic confidence/
correctness lists — no dataset or trained model needed.

Run:
    uv run pytest tests/test_selective.py
"""

from pet_breed_classification.helpers.selective import (
    compute_selective_metrics,
    find_threshold_for_target_accuracy,
)

# 10 examples: confidence and whether the prediction was correct.
# Higher-confidence examples are deliberately more often correct, so a
# higher threshold should trade coverage for accuracy.
CONFIDENCES = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.55, 0.50, 0.40]
CORRECT =     [True, True, True, True, False, True, False, False, True, False]


def test_compute_selective_metrics_at_known_threshold():
    # threshold=0.80 accepts the first 5 examples (0.99..0.80): 4 correct, 1 wrong
    metrics = compute_selective_metrics(CONFIDENCES, CORRECT, threshold=0.80)

    assert metrics["threshold"] == 0.80
    assert metrics["coverage"] == 5 / 10
    assert metrics["selective_accuracy"] == 4 / 5
    assert metrics["overall_accuracy"] == sum(CORRECT) / len(CORRECT)


def test_higher_threshold_never_increases_coverage():
    low = compute_selective_metrics(CONFIDENCES, CORRECT, threshold=0.5)
    high = compute_selective_metrics(CONFIDENCES, CORRECT, threshold=0.9)

    assert high["coverage"] <= low["coverage"]


def test_threshold_of_zero_accepts_everything():
    metrics = compute_selective_metrics(CONFIDENCES, CORRECT, threshold=0.0)

    assert metrics["coverage"] == 1.0
    assert metrics["selective_accuracy"] == metrics["overall_accuracy"]


def test_find_threshold_meets_target_accuracy():
    threshold = find_threshold_for_target_accuracy(CONFIDENCES, CORRECT, target_accuracy=0.95)
    metrics = compute_selective_metrics(CONFIDENCES, CORRECT, threshold)

    assert metrics["selective_accuracy"] >= 0.95


def test_find_threshold_returns_1_when_target_unreachable():
    # No subset of this data reaches 100% accuracy with more than 0 examples
    # accepted except possibly a single perfect point; use an impossible target.
    threshold = find_threshold_for_target_accuracy(CONFIDENCES, CORRECT, target_accuracy=1.5)
    assert threshold == 1.0
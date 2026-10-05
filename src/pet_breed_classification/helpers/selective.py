"""
Abstention threshold selection and selective-accuracy metrics.

Pure functions operating on (confidence, correct) pairs already computed
by helpers/calibration.py — no model, no I/O here. "Accepted" means the
model's calibrated confidence is >= threshold; everything else is treated
as abstained ("uncertain").
"""


def compute_selective_metrics(
    confidences: list[float], correct: list[bool], threshold: float
) -> dict:
    """Metrics for one candidate threshold.

    coverage            = fraction of examples accepted (confidence >= threshold)
    selective_accuracy   = accuracy among accepted examples only
    overall_accuracy     = accuracy on ALL examples, ignoring abstention
                            (i.e. what accuracy would be with no abstention at all)
    """
    overall_accuracy = sum(correct) / len(correct)

    accepted_correct = [
        is_correct for conf, is_correct in zip(confidences, correct) if conf >= threshold
    ]
    coverage = len(accepted_correct) / len(correct)
    selective_accuracy = sum(accepted_correct) / len(accepted_correct) if accepted_correct else 0.0

    return {
        "threshold": threshold,
        "coverage": coverage,
        "selective_accuracy": selective_accuracy,
        "overall_accuracy": overall_accuracy,
    }


def find_threshold_for_target_accuracy(
    confidences: list[float], correct: list[bool], target_accuracy: float
) -> float:
    """Smallest threshold whose selective accuracy reaches target_accuracy
    on the given (validation) set — maximizes coverage subject to hitting
    the accuracy bar. Candidate thresholds are the sorted, unique observed
    confidence values, so every "interesting" cutoff is checked.

    If no threshold reaches the target (the model is never that reliable),
    returns 1.0 — the strictest possible threshold, abstaining on nearly
    everything, which is the honest answer in that case.
    """
    candidate_thresholds = sorted(set(confidences))

    for threshold in candidate_thresholds:
        metrics = compute_selective_metrics(confidences, correct, threshold)
        if metrics["selective_accuracy"] >= target_accuracy:
            return threshold

    return 1.0

# ---------------------------------------------------------------------------
# Metrics (plain Python/torch — no extra dependency)
# ---------------------------------------------------------------------------

from typing import List


def compute_top1_accuracy(all_preds: List[int], all_labels: List[int]) -> float:
    correct = sum(p == y for p, y in zip(all_preds, all_labels))
    return correct / len(all_labels)


def compute_macro_f1(all_preds: List[int], all_labels: List[int], num_classes: int) -> float:
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


def compute_ece(all_confidences: List[float], all_correct: List[bool], num_bins: int) -> float:
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


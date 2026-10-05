"""
Confusion matrix and most-confused-pairs utilities.

Pure functions on plain int lists — no torch, no I/O, no model. Works
directly on predicted class indices, which temperature scaling never
changes (dividing logits by a positive scalar preserves argmax), so this
can reuse whatever logits a calibration run already computed.
"""


def build_confusion_matrix(
    true_labels: list[int], predicted_labels: list[int], num_classes: int
) -> list[list[int]]:
    """Returns a num_classes x num_classes count matrix where
    matrix[true][predicted] = number of examples with that true/predicted
    class pair. The diagonal is correct predictions; everything off the
    diagonal is a confusion.
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_idx, pred_idx in zip(true_labels, predicted_labels):
        matrix[true_idx][pred_idx] += 1
    return matrix


def most_confused_pairs(
    confusion_matrix: list[list[int]],
    index_to_breed: dict[int, str],
    top_n: int,
) -> list[tuple[str, str, int]]:
    """Returns the top_n (true_breed, predicted_breed, count) entries with
    the highest off-diagonal counts — the project's most commonly confused
    breed pairs, sorted by count descending.
    """
    off_diagonal_counts = []
    num_classes = len(confusion_matrix)

    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = confusion_matrix[true_idx][pred_idx]
            if count > 0:
                off_diagonal_counts.append(
                    (index_to_breed[true_idx], index_to_breed[pred_idx], count)
                )

    off_diagonal_counts.sort(key=lambda entry: entry[2], reverse=True)
    return off_diagonal_counts[:top_n]
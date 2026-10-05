"""
Unit tests for helpers/confusion.py, using small synthetic label lists —
no dataset, model, or torch needed.

Run:
    uv run pytest tests/test_confusion.py
"""

from pet_breed_classification.helpers.confusion import (
    build_confusion_matrix,
    most_confused_pairs,
)

# 3 classes: 0=cat, 1=dog, 2=bird
# True labels and what the model predicted for each example.
TRUE_LABELS = [0, 0, 0, 1, 1, 1, 2, 2]
PREDICTED =   [0, 0, 1, 1, 1, 0, 2, 0]
INDEX_TO_BREED = {0: "cat", 1: "dog", 2: "bird"}


def test_build_confusion_matrix_correct_counts():
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED, num_classes=3)

    assert matrix[0][0] == 2  # true=cat, pred=cat
    assert matrix[0][1] == 1  # true=cat, pred=dog
    assert matrix[1][1] == 2  # true=dog, pred=dog
    assert matrix[1][0] == 1  # true=dog, pred=cat
    assert matrix[2][2] == 1  # true=bird, pred=bird
    assert matrix[2][0] == 1  # true=bird, pred=cat


def test_confusion_matrix_total_equals_num_examples():
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED, num_classes=3)
    total = sum(sum(row) for row in matrix)
    assert total == len(TRUE_LABELS)


def test_most_confused_pairs_excludes_diagonal():
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED, num_classes=3)
    pairs = most_confused_pairs(matrix, INDEX_TO_BREED, top_n=10)

    for true_breed, predicted_breed, _count in pairs:
        assert true_breed != predicted_breed


def test_most_confused_pairs_sorted_descending():
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED, num_classes=3)
    pairs = most_confused_pairs(matrix, INDEX_TO_BREED, top_n=10)
    counts = [count for _, _, count in pairs]

    assert counts == sorted(counts, reverse=True)


def test_most_confused_pairs_respects_top_n():
    matrix = build_confusion_matrix(TRUE_LABELS, PREDICTED, num_classes=3)
    pairs = most_confused_pairs(matrix, INDEX_TO_BREED, top_n=1)

    assert len(pairs) == 1
    # cat<->dog confusions (count=1 each) and bird->cat (count=1) are tied;
    # just confirm the single returned entry is a real off-diagonal count.
    assert pairs[0][2] >= 1
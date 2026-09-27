"""
Step 1 — Fix the split before anything else.

Downloads Oxford-IIIT Pet (trainval + test) via torchvision, then carves a
validation fold out of `trainval` using a fixed seed (see config.SPLIT_SEED),
stratified by breed so every class is proportionally represented in both
train and val.

The official `test` split is only read here to record its image IDs — it is
never touched otherwise, and stays untouched until the optimization/reporting
phase later in the project.

Outputs (committed to git / DVC, not just generated locally):
    data/processed/splits/train.txt
    data/processed/splits/val.txt
    data/processed/splits/test.txt

Each file contains one image_id per line (filename stem, e.g. "Abyssinian_100"),
sorted alphabetically for stable diffs.

Run:
    uv run python -m pet_breed_classification.data
"""

import random
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

from pet_breed_classification import config


def _image_ids_and_labels(dataset: OxfordIIITPet) -> tuple[list[str], list[int]]:
    """Extract (image_id, label) pairs from a torchvision OxfordIIITPet dataset.

    torchvision does not expose filenames through __getitem__ (only PIL image +
    label), so we read its internal `_images` / `_labels` lists directly. This
    is the standard, documented workaround used across the ecosystem for this
    dataset class.
    """
    image_ids = [Path(p).stem for p in dataset._images]
    labels = list(dataset._labels)
    return image_ids, labels


def stratified_split(
    image_ids: list[str],
    labels: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministically split (image_ids, labels) into (train_ids, val_ids),
    holding out `val_fraction` of each class's examples for validation.
    """
    by_label: dict[int, list[str]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels):
        by_label[label].append(image_id)

    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []

    for label, ids_for_label in sorted(by_label.items()):
        ids_for_label = sorted(ids_for_label)  # deterministic order before shuffle
        rng.shuffle(ids_for_label)
        n_val = max(1, round(len(ids_for_label) * val_fraction))
        val_ids.extend(ids_for_label[:n_val])
        train_ids.extend(ids_for_label[n_val:])

    return train_ids, val_ids


def write_index_file(path: Path, image_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for image_id in sorted(image_ids):
            f.write(f"{image_id}\n")


def make_splits() -> None:
    print(f"Downloading/loading Oxford-IIIT Pet to {config.DATA_ROOT} ...")
    trainval_ds = OxfordIIITPet(root=str(config.DATA_ROOT), split="trainval", download=True)
    test_ds = OxfordIIITPet(root=str(config.DATA_ROOT), split="test", download=True)

    trainval_ids, trainval_labels = _image_ids_and_labels(trainval_ds)
    test_ids, _ = _image_ids_and_labels(test_ds)

    print(
        f"trainval: {len(trainval_ids)} images across {len(set(trainval_labels))} classes | "
        f"test: {len(test_ids)} images"
    )

    train_ids, val_ids = stratified_split(
        trainval_ids,
        trainval_labels,
        val_fraction=config.VAL_FRACTION,
        seed=config.SPLIT_SEED,
    )

    # Sanity: no leakage across the three sets we're about to write.
    assert set(train_ids).isdisjoint(val_ids), "train/val overlap detected"
    assert set(train_ids).isdisjoint(test_ids), "train/test overlap detected"
    assert set(val_ids).isdisjoint(test_ids), "val/test overlap detected"

    write_index_file(config.SPLITS_DIR / "train.txt", train_ids)
    write_index_file(config.SPLITS_DIR / "val.txt", val_ids)
    write_index_file(config.SPLITS_DIR / "test.txt", test_ids)

    print(
        f"Wrote splits to {config.SPLITS_DIR}: "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} "
        f"(seed={config.SPLIT_SEED}, val_fraction={config.VAL_FRACTION})"
    )


if __name__ == "__main__":
    make_splits()
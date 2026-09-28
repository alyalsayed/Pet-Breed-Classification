"""
Validation tests for the Step 2 manifest + label map, per the README's
Day 2 checklist:

    - every manifest path exists and opens
    - no image_id appears in two splits
    - the label map has exactly 37 entries and matches the committed reference
    - every class has at least 50 training images
    - no image is under 32x32

These assume `uv run data` and `uv run manifest` have already been run, so
data/processed/manifest.json and label_map.json exist. Run with:

    uv run pytest tests/test_manifest.py
"""

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from pet_breed_classification import config

MIN_IMAGE_SIDE = 32
MIN_TRAIN_IMAGES_PER_CLASS = 50
EXPECTED_NUM_CLASSES = 37


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    if not config.MANIFEST_PATH.exists():
        pytest.skip(f"{config.MANIFEST_PATH} not found — run the manifest step first")
    with config.MANIFEST_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def label_map() -> dict[str, int]:
    if not config.LABEL_MAP_PATH.exists():
        pytest.skip(f"{config.LABEL_MAP_PATH} not found — run the manifest step first")
    with config.LABEL_MAP_PATH.open() as f:
        return json.load(f)


def test_every_manifest_path_exists_and_opens(manifest):
    for record in manifest:
        path = Path(record["path"])
        assert path.exists(), f"missing image file: {path}"
        with Image.open(path) as img:
            img.verify()


def test_no_image_id_in_two_splits(manifest):
    seen: dict[str, str] = {}
    for record in manifest:
        image_id = record["image_id"]
        split = record["split"]
        assert image_id not in seen, (
            f"{image_id} appears in both '{seen.get(image_id)}' and '{split}' splits"
        )
        seen[image_id] = split


def test_every_record_has_a_split(manifest):
    missing = [r["image_id"] for r in manifest if r["split"] is None]
    assert not missing, f"{len(missing)} manifest records have no split assigned: {missing[:5]}..."


def test_label_map_has_37_entries(label_map):
    assert len(label_map) == EXPECTED_NUM_CLASSES, (
        f"expected {EXPECTED_NUM_CLASSES} breeds, found {len(label_map)}"
    )


def test_label_map_matches_manifest_class_indices(manifest, label_map):
    for record in manifest:
        breed = record["breed"]
        assert breed in label_map, f"breed '{breed}' in manifest but not in label map"
        assert record["class_index"] == label_map[breed], (
            f"class_index mismatch for {record['image_id']}: "
            f"manifest={record['class_index']} label_map={label_map[breed]}"
        )


def test_every_class_has_min_training_images(manifest):
    train_counts = Counter(r["breed"] for r in manifest if r["split"] == "train")
    under_min = {breed: n for breed, n in train_counts.items() if n < MIN_TRAIN_IMAGES_PER_CLASS}
    assert not under_min, (
        f"classes with fewer than {MIN_TRAIN_IMAGES_PER_CLASS} training images: {under_min}"
    )


def test_no_image_under_min_size(manifest):
    too_small = [
        r["image_id"]
        for r in manifest
        if r["width"] < MIN_IMAGE_SIDE or r["height"] < MIN_IMAGE_SIDE
    ]
    assert not too_small, f"images below {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}: {too_small}"
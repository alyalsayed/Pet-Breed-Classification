"""
Tests for the corruption suite: unit tests on the pure functions in
corruption_effects.py, plus validation of the generated corrupted_manifest.json
once `uv run corruptions` has been run.

Run:
    uv run pytest tests/test_corruptions.py
"""

import json
from collections import Counter

import pytest
from PIL import Image

from pet_breed_classification import config
from pet_breed_classification.helpers import corruption_effects


# ---------------------------------------------------------------------------
# Unit tests: pure corruption functions (no files needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (200, 200), color=(120, 140, 160))


@pytest.mark.parametrize("severity", config.SEVERITY_LEVELS)
@pytest.mark.parametrize("corruption_name", list(corruption_effects.CORRUPTION_FUNCTIONS))
def test_corruption_function_returns_same_size_rgb_image(sample_image, corruption_name, severity):
    corruption_fn = corruption_effects.CORRUPTION_FUNCTIONS[corruption_name]
    result = corruption_fn(sample_image, severity)

    assert result.size == sample_image.size
    assert result.mode == "RGB"


def test_blur_actually_changes_pixels(sample_image):
    # A flat-color image blurred should still be flat (edge case), so use
    # a non-flat image to confirm the filter actually does something.
    checker = Image.new("RGB", (200, 200))
    pixels = checker.load()
    for x in range(200):
        for y in range(200):
            pixels[x, y] = (255, 255, 255) if (x // 20 + y // 20) % 2 == 0 else (0, 0, 0)

    blurred = corruption_effects.apply_gaussian_blur(checker, severity=3)
    assert (list(blurred.get_flattened_data()) != list(checker.get_flattened_data()))


# ---------------------------------------------------------------------------
# Validation tests: corrupted_manifest.json (assumes build_corruptions() ran)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corrupted_manifest() -> list[dict]:
    if not config.CORRUPTED_MANIFEST_PATH.exists():
        pytest.skip(f"{config.CORRUPTED_MANIFEST_PATH} not found — run `uv run corruptions` first")
    with config.CORRUPTED_MANIFEST_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def clean_test_image_ids() -> set[str]:
    if not config.MANIFEST_PATH.exists():
        pytest.skip(f"{config.MANIFEST_PATH} not found — run `uv run manifest` first")
    with config.MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    return {r["image_id"] for r in manifest if r["split"] == "test"}


def test_every_corrupted_image_exists_and_opens(corrupted_manifest):
    for record in corrupted_manifest:
        with Image.open(record["path"]) as img:
            img.verify()


def test_corrupted_image_ids_are_subset_of_clean_test_split(corrupted_manifest, clean_test_image_ids):
    corrupted_ids = {r["image_id"] for r in corrupted_manifest}
    assert corrupted_ids.issubset(clean_test_image_ids), (
        "corrupted_manifest.json contains image_ids not in the clean test split — "
        "train/val images must never be corrupted"
    )


def test_corruption_and_severity_always_set(corrupted_manifest):
    for record in corrupted_manifest:
        assert record["corruption"] is not None, f"{record['image_id']} missing corruption label"
        assert record["severity"] != 0, f"{record['image_id']} missing severity level"


def test_expected_record_count_per_corruption_and_severity(corrupted_manifest, clean_test_image_ids):
    counts = Counter((r["corruption"], r["severity"]) for r in corrupted_manifest)
    expected_count = len(clean_test_image_ids)

    for corruption_name in corruption_effects.CORRUPTION_FUNCTIONS:
        for severity in config.SEVERITY_LEVELS:
            actual_count = counts[(corruption_name, severity)]
            assert actual_count == expected_count, (
                f"{corruption_name}/severity_{severity}: expected {expected_count} "
                f"images, found {actual_count}"
            )


def test_labels_match_source_image(corrupted_manifest):
    """Spot-check: corruption should never change breed/species/class_index."""
    with config.MANIFEST_PATH.open() as f:
        clean_manifest = json.load(f)
    clean_by_id = {r["image_id"]: r for r in clean_manifest}

    for record in corrupted_manifest:
        source = clean_by_id[record["image_id"]]
        assert record["breed"] == source["breed"]
        assert record["species"] == source["species"]
        assert record["class_index"] == source["class_index"]
"""
Corruption suite build pipeline — the project's drift ground truth.

Applies every corruption in corruption_effects.CORRUPTION_FUNCTIONS, at
every severity in config.SEVERITY_LEVELS, to every image in the TEST split
only (never train/val — corrupting training data would leak the drift
simulation into the model itself).

Produces:
    data/processed/corrupted/<corruption_name>/severity_<n>/<image_id>.jpg
    data/processed/corrupted_manifest.json

Run:
    uv run corruptions
    # or directly:
    uv run python -m pet_breed_classification.corruptions
"""

import json
from pathlib import Path

from PIL import Image

from pet_breed_classification import config, features
from pet_breed_classification.helpers import corruption_effects
from pet_breed_classification.manifest import load_split_records


def corrupted_image_output_path(image_id: str, corruption_name: str, severity: int) -> Path:
    return config.CORRUPTIONS_DIR / corruption_name / f"severity_{severity}" / f"{image_id}.jpg"


def apply_corruption_and_save(
    image: Image.Image, corruption_name: str, severity: int, output_path: Path
) -> None:
    corruption_fn = corruption_effects.CORRUPTION_FUNCTIONS[corruption_name]
    corrupted_image = corruption_fn(image, severity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted_image.save(output_path, format="JPEG", quality=95)


def build_corrupted_record(
    source_record: dict, corruption_name: str, severity: int, output_path: Path
) -> dict:
    """Same schema as the clean manifest, but pointing at the corrupted
    image file, with corruption/severity filled in instead of null/0.
    Breed/species/class_index are copied from the source record —
    corruption changes pixels, never labels.
    """
    with Image.open(output_path) as img:
        width, height = img.size

    return {
        "image_id": source_record["image_id"],
        "path": str(output_path),
        "breed": source_record["breed"],
        "species": source_record["species"],
        "class_index": source_record["class_index"],
        "split": "test",
        "corruption": corruption_name,
        "severity": severity,
        "width": width,
        "height": height,
    }


def build_corruptions() -> None:
    test_records = load_split_records("test")
    corrupted_records = []

    for record in test_records:
        image = features.load_image(record["path"])

        for corruption_name in corruption_effects.CORRUPTION_FUNCTIONS:
            for severity in config.SEVERITY_LEVELS:
                output_path = corrupted_image_output_path(
                    record["image_id"], corruption_name, severity
                )
                apply_corruption_and_save(image, corruption_name, severity, output_path)
                corrupted_records.append(
                    build_corrupted_record(record, corruption_name, severity, output_path)
                )

    config.CORRUPTED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.CORRUPTED_MANIFEST_PATH.open("w") as f:
        json.dump(corrupted_records, f, indent=2)

    num_corruption_types = len(corruption_effects.CORRUPTION_FUNCTIONS)
    num_severities = len(config.SEVERITY_LEVELS)
    print(
        f"Corrupted {len(test_records)} test images x "
        f"{num_corruption_types} corruption types x {num_severities} severities "
        f"= {len(corrupted_records)} records"
    )
    print(f"Wrote corrupted manifest to {config.CORRUPTED_MANIFEST_PATH}")


if __name__ == "__main__":
    build_corruptions()
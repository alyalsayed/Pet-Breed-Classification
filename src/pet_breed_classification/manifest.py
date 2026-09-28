"""
Manifest generation and label-map generation for the Pet Breed
Classification project.

Reads the split index files written by data.py, plus the dataset's own
annotations/list.txt, and produces:
    - data/processed/manifest.json  (one record per image)
    - data/processed/label_map.json (our own sorted, committed breed -> index map)

Run:
    uv run manifest
    # or directly:
    uv run python -m pet_breed_classification.manifest
"""

import json
from pathlib import Path

from PIL import Image

from pet_breed_classification import config


def _parse_species_by_image_id(list_txt_path: Path) -> dict[str, str]:
    """Parse annotations/list.txt.

    Format per line: `Image CLASS-ID SPECIES BREED-ID` (species: 1=Cat, 2=Dog).
    Returns {image_id: "cat" | "dog"}.

    We use this file only for the species field — CLASS-ID is intentionally
    ignored as a label source (see build_label_map): the README is explicit
    that class indices must come from our own committed, sorted label map,
    never from an externally-provided ordering.
    """
    species_by_id: dict[str, str] = {}
    with list_txt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image_id, _class_id, species_code, _breed_id = line.split()
            species_by_id[image_id] = "cat" if species_code == "1" else "dog"
    return species_by_id


def _breed_name_from_image_id(image_id: str) -> str:
    """Oxford-IIIT Pet filename convention: '<Breed_Name>_<index>' -> breed name."""
    return image_id.rsplit("_", 1)[0]


def _load_split_membership(splits_dir: Path) -> dict[str, str]:
    """Read the Step 1 split files and invert them into {image_id: split_name}.

    Raises if any image_id appears in more than one split file — this is the
    same leakage guard as in data.py, re-applied here since the manifest is
    the artifact that training/serving code will actually trust.
    """
    membership: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_file = splits_dir / f"{split_name}.txt"
        for image_id in split_file.read_text().splitlines():
            if image_id in membership:
                raise ValueError(f"{image_id} appears in multiple split files")
            membership[image_id] = split_name
    return membership


def build_label_map(breed_names: set[str]) -> dict[str, int]:
    """Our own sorted, committed label map — the single source of truth for
    class_index everywhere downstream (training, serving, calibration).
    """
    return {breed: idx for idx, breed in enumerate(sorted(breed_names))}


def build_manifest() -> None:
    species_by_id = _parse_species_by_image_id(config.ANNOTATIONS_LIST_PATH)
    split_by_id = _load_split_membership(config.SPLITS_DIR)

    image_ids = sorted(species_by_id.keys())
    breed_names = {_breed_name_from_image_id(image_id) for image_id in image_ids}
    label_map = build_label_map(breed_names)

    records = []
    for image_id in image_ids:
        if image_id not in split_by_id:
            raise ValueError(
                f"{image_id} is in annotations/list.txt but missing from every "
                f"split file in {config.SPLITS_DIR} — run `uv run data` first, "
                f"or check that the split files are complete."
            )

        breed = _breed_name_from_image_id(image_id)
        image_path = config.IMAGES_DIR / f"{image_id}.jpg"

        with Image.open(image_path) as img:
            width, height = img.size

        records.append(
            {
                "image_id": image_id,
                "path": str(image_path),
                "breed": breed,
                "species": species_by_id[image_id],
                "class_index": label_map[breed],
                "split": split_by_id[image_id],
                "corruption": None,
                "severity": 0,
                "width": width,
                "height": height,
            }
        )

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.MANIFEST_PATH.open("w") as f:
        json.dump(records, f, indent=2)

    config.LABEL_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.LABEL_MAP_PATH.open("w") as f:
        json.dump(label_map, f, indent=2, sort_keys=True)

    print(f"Wrote manifest with {len(records)} records to {config.MANIFEST_PATH}")
    print(f"Wrote label map with {len(label_map)} breeds to {config.LABEL_MAP_PATH}")


if __name__ == "__main__":
    build_manifest()
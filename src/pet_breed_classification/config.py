"""
Centralized configuration for the Pet Breed Classification project.

Single source of truth for filesystem paths and reproducibility constants.
As the project grows, add new config values here rather than hardcoding
them in individual modules — e.g. corruption-suite output paths (Day 3),
training hyperparameters (Day 4+).
"""

from pathlib import Path

# --- Repo-root-relative filesystem layout: raw data ---
DATA_ROOT = Path("data/raw")
RAW_DATASET_DIR = DATA_ROOT / "oxford-iiit-pet"
IMAGES_DIR = RAW_DATASET_DIR / "images"
ANNOTATIONS_LIST_PATH = RAW_DATASET_DIR / "annotations" / "list.txt"

# --- Repo-root-relative filesystem layout: processed / committed artifacts ---
SPLITS_DIR = Path("data/processed/splits")
MANIFEST_PATH = Path("data/processed/manifest.json")
LABEL_MAP_PATH = Path("data/processed/label_map.json")

# --- Reproducibility contract ---
SPLIT_SEED = 42
VAL_FRACTION = 0.15
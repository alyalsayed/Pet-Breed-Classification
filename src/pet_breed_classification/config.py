"""
Centralized configuration for the Pet Breed Classification project.

Single source of truth for filesystem paths and reproducibility constants.
As the project grows, add new config values here rather than hardcoding
them in individual modules — e.g. manifest/label-map paths (Day 2),
corruption-suite output paths (Day 3), training hyperparameters (Day 4+).
"""

from pathlib import Path

# --- Repo-root-relative filesystem layout ---
DATA_ROOT = Path("data/raw")
SPLITS_DIR = Path("data/processed/splits")

# --- Reproducibility contract ---
SPLIT_SEED = 42
VAL_FRACTION = 0.15
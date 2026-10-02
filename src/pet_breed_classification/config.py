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

# --- Shared preprocessing (features.py) ---
RESIZE_SIZE = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Training (train.py) ---
NUM_CLASSES = 37
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
ECE_NUM_BINS = 10

MLFLOW_EXPERIMENT_NAME = "pet-breed-classification"
CHECKPOINT_DIR = Path("models")

# --- Corruption suite (corruption_effects.py, corruptions.py) ---
SEVERITY_LEVELS = (1, 2, 3)

CORRUPTIONS_DIR = Path("data/processed/corrupted")
CORRUPTED_MANIFEST_PATH = Path("data/processed/corrupted_manifest.json")

# Parameters per severity level (1=mild, 2=moderate, 3=severe). Severity 3
# matches the handbook's named examples (JPEG quality 30, downscale to
# 96x96); levels 1-2 are our own reasonable interpolation, since the
# handbook only specifies one example value per corruption type.
BLUR_RADIUS_BY_SEVERITY = {1: 1, 2: 2, 3: 4}
BRIGHTNESS_UP_FACTOR_BY_SEVERITY = {1: 1.3, 2: 1.6, 3: 2.0}
BRIGHTNESS_DOWN_FACTOR_BY_SEVERITY = {1: 0.7, 2: 0.5, 3: 0.3}
JPEG_QUALITY_BY_SEVERITY = {1: 50, 2: 40, 3: 30}
DOWNSCALE_SIZE_BY_SEVERITY = {1: 128, 2: 96, 3: 64}
MOTION_BLUR_KERNEL_SIZE_BY_SEVERITY = {1: 5, 2: 9, 3: 15}

# --- Reproducibility contract ---
SPLIT_SEED = 42
VAL_FRACTION = 0.15
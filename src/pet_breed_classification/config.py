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

# --- Image preprocessing ---
RESIZE_SIZE = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Training (train.py) ---
SUPPORTED_BACKBONES = ("resnet50", "resnet18", "mobilenet_v3_small")
MODEL_VERSION = "v1"
DATASET_NAME = "oxford-iiit-pet"
NUM_CLASSES = 37
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
ECE_NUM_BINS = 10

MLFLOW_EXPERIMENT_NAME = "pet-breed-classification"
CHECKPOINT_DIR = Path("models")

# --- Corruption suite (corruption_effects.py, corruptions.py) ---
SEVERITY_LEVELS = (1, 2, 3)

CORRUPTIONS_DIR = Path("data/processed/corrupted")
CORRUPTED_MANIFEST_PATH = Path("data/processed/corrupted_manifest.json")

# --- Corruption hyperparameters (corruptions.py) ---
BLUR_RADIUS_BY_SEVERITY = {1: 1, 2: 2, 3: 4}
BRIGHTNESS_UP_FACTOR_BY_SEVERITY = {1: 1.3, 2: 1.6, 3: 2.0}
BRIGHTNESS_DOWN_FACTOR_BY_SEVERITY = {1: 0.7, 2: 0.5, 3: 0.3}
JPEG_QUALITY_BY_SEVERITY = {1: 50, 2: 40, 3: 30}
DOWNSCALE_SIZE_BY_SEVERITY = {1: 128, 2: 96, 3: 64}
MOTION_BLUR_KERNEL_SIZE_BY_SEVERITY = {1: 5, 2: 9, 3: 15}

# --- Calibration (helpers/calibration.py, calibrate.py) ---
CALIBRATION_RESULTS_PATH = Path("models/resnet50_best_calibration.json")
CALIBRATION_REPORT_PATH = Path("reports/calibration.png")

# --- Temperature scaling (helpers/calibration.py) ---
TEMPERATURE_INIT = 1.0
TEMPERATURE_LR = 0.01
TEMPERATURE_NUM_STEPS = 200

# --- Abstention threshold (helpers/selective.py) ---
TARGET_SELECTIVE_ACCURACY = 0.95

# --- Confusion matrix (helpers/confusion.py) ---
CONFUSION_MATRIX_REPORT_PATH = Path("reports/confusion_matrix.png")
TOP_CONFUSED_PAIRS_COUNT = 5

# --- Confusion matrix (helpers/confusion.py) ---
CONFUSION_MATRIX_REPORT_PATH = Path("reports/confusion_matrix.png")
TOP_CONFUSED_PAIRS_COUNT = 5

# --- Reproducibility contract ---
SPLIT_SEED = 42
VAL_FRACTION = 0.15
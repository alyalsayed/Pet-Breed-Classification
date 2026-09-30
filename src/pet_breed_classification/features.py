"""
Shared preprocessing for the Pet Breed Classification project.

This is the single place the eval/inference transform is defined. Training
(train.py), validation, and — later — the serving API must all import
get_eval_transform() and load_image() from here rather than redefining
their own preprocessing. This is what prevents train/serve preprocessing
skew, the most common silent failure mode for this kind of project.

Augmentation (random crop/flip, etc.) is intentionally NOT here — it is
training-only and belongs in train.py, applied on top of this transform,
never mixed into it.
"""

from pathlib import Path

from PIL import Image
from torchvision import transforms

from pet_breed_classification import config


def get_eval_transform() -> transforms.Compose:
    """The single, deterministic preprocessing transform for the model.

    Used for training-time evaluation, validation, testing, and — once
    exported alongside the trained weights — for serving. No randomness.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def load_image(path: Path) -> Image.Image:
    """Load an image and force RGB.

    The dataset contains a mix of image modes (some CMYK, some greyscale,
    at least one RGBA/PNG with an alpha channel). Converting to RGB here,
    in the one shared loading path, keeps training and serving consistent
    regardless of the source image's original mode.
    """
    with Image.open(path) as img:
        return img.convert("RGB")
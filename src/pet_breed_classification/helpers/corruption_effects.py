"""
Pure image corruption functions.

Each function takes a PIL image and a severity level (1=mild, 2=moderate,
3=severe) and returns a new, corrupted PIL image. No file I/O, no manifest
logic, no project-wide orchestration here — that lives in corruptions.py.
Keeping these functions pure makes them trivial to test in isolation.

Severity -> parameter values come from config.py, never hardcoded here.
"""

import io

from PIL import Image, ImageEnhance, ImageFilter

from pet_breed_classification import config


def apply_gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    """Simulates an out-of-focus phone camera."""
    radius = config.BLUR_RADIUS_BY_SEVERITY[severity]
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness_up(image: Image.Image, severity: int) -> Image.Image:
    """Simulates a photo taken in direct sunlight (overexposed)."""
    factor = config.BRIGHTNESS_UP_FACTOR_BY_SEVERITY[severity]
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_brightness_down(image: Image.Image, severity: int) -> Image.Image:
    """Simulates a dim indoor-evening photo (underexposed)."""
    factor = config.BRIGHTNESS_DOWN_FACTOR_BY_SEVERITY[severity]
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_jpeg_compression(image: Image.Image, severity: int) -> Image.Image:
    """Simulates a messaging-app re-encode: compress to JPEG in memory,
    then reload it, so the quality loss is actually "baked in" to the
    returned pixels rather than only applied once saved to disk later.
    """
    quality = config.JPEG_QUALITY_BY_SEVERITY[severity]
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_downscale_upscale(image: Image.Image, severity: int) -> Image.Image:
    """Simulates an old handset's low-resolution camera: shrink, then
    stretch back to the original size, losing detail in the process.
    """
    target_size = config.DOWNSCALE_SIZE_BY_SEVERITY[severity]
    original_size = image.size
    small = image.resize((target_size, target_size), Image.BILINEAR)
    return small.resize(original_size, Image.BILINEAR)


def apply_motion_blur(image: Image.Image, severity: int) -> Image.Image:
    """Simulates camera/subject motion during the shot, via a simple
    horizontal averaging kernel (no extra dependency like OpenCV needed).
    """
    kernel_size = config.MOTION_BLUR_KERNEL_SIZE_BY_SEVERITY[severity]
    kernel_values = [1 / kernel_size] * kernel_size
    motion_kernel = ImageFilter.Kernel(size=(kernel_size, 1), kernel=kernel_values)
    return image.filter(motion_kernel)


# Single registry every other module uses to iterate corruption types —
# add a new corruption here and it's automatically picked up everywhere.
CORRUPTION_FUNCTIONS = {
    "gaussian_blur": apply_gaussian_blur,
    "brightness_up": apply_brightness_up,
    "brightness_down": apply_brightness_down,
    "jpeg_compression": apply_jpeg_compression,
    "downscale_upscale": apply_downscale_upscale,
    "motion_blur": apply_motion_blur,
}
"""
Quick manual test: load the trained model (v1) from models/ and run it on
one sample test image, printing whether the prediction matches the true
breed.

This is a lightweight sanity check, not a full evaluation — for real
metrics (accuracy, macro-F1, ECE) see train.py's validation loop.

Run:
    uv run predict
    # or directly:
    uv run python -m pet_breed_classification.main
"""

import json

import torch

from . import config, features
from .helpers.checkpoint import load_checkpoint
from .helpers.model import build_model, get_device
from .manifest import load_split_records

MODEL_VERSION = "v1"
BACKBONE_NAME = "resnet50"


def load_index_to_breed() -> dict[int, str]:
    """Invert label_map.json (breed -> index) into (index -> breed), so we
    can turn the model's predicted class index back into a breed name.
    """
    with config.LABEL_MAP_PATH.open() as f:
        label_map = json.load(f)
    return {index: breed for breed, index in label_map.items()}


def predict_single_image(model, transform, image_path, device) -> tuple[int, float]:
    """Run the model on one image file and return (predicted_index, confidence)."""
    image = features.load_image(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probs, dim=1)

    return predicted_index.item(), confidence.item()


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")
    print(f"Testing model version: {MODEL_VERSION} ({BACKBONE_NAME})")

    model = build_model().to(device)
    load_checkpoint(model, BACKBONE_NAME, device)

    index_to_breed = load_index_to_breed()
    transform = features.get_eval_transform()

    # Pick one sample test image — the first record in the test split.
    test_records = load_split_records("test")
    sample = test_records[0]

    predicted_index, confidence = predict_single_image(model, transform, sample["path"], device)
    predicted_breed = index_to_breed[predicted_index]
    true_breed = sample["breed"]

    print(f"Image:           {sample['image_id']}")
    print(f"True breed:      {true_breed}")
    print(f"Predicted breed: {predicted_breed} (confidence={confidence:.2f})")
    print("Result:", "CORRECT" if predicted_breed == true_breed else "WRONG")


if __name__ == "__main__":
    main()
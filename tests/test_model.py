"""
Unit tests for helpers/model.py's build_model(). CPU-only, no dataset,
no GPU needed — just confirms each supported backbone constructs
correctly and outputs the right shape before spending Colab GPU time
on it.

Run:
    uv run pytest tests/test_model.py
"""

import pytest
import torch

from pet_breed_classification import config
from pet_breed_classification.helpers.model import build_model

BATCH_SIZE = 2


@pytest.mark.parametrize("backbone_name", config.SUPPORTED_BACKBONES)
def test_build_model_outputs_correct_shape(backbone_name):
    model = build_model(backbone_name)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (BATCH_SIZE, config.NUM_CLASSES)


def test_build_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        build_model("not_a_real_backbone")
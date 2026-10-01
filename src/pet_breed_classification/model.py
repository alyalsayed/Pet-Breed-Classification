
from torchvision.models import ResNet50_Weights, resnet50

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """CUDA if available (e.g. Colab), otherwise CPU (e.g. local WSL) —
    same code runs unchanged in both places.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """ResNet-50 with ImageNet-pretrained weights, explicit and simple:
    the exact weight set is named directly, and torchvision downloads it
    to its local cache automatically the first time this runs — no extra
    download logic needed.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    return model

"""
Temperature scaling for model calibration.

Pure functions operating on already-computed logits/labels — no file I/O,
no model loading, no dataloaders here. That orchestration lives in
calibrate.py. This separation is what makes temperature fitting and ECE
math easy to unit-test with small synthetic examples.

Background: a trained classifier's raw softmax confidence is usually
overconfident. Temperature scaling divides the logits by a single learned
scalar T (> 1 makes the model less confident) before taking softmax. It
never changes which class is predicted — only how confident the model
claims to be — so accuracy is identical before and after; only ECE changes.
"""

import torch
from torch import nn

from .metrics import compute_ece
from .. import config


@torch.no_grad()
def collect_logits_and_labels(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model once over `loader` and collect raw logits (pre-softmax)
    and true labels for every example. Used once on the validation set.
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        all_logits.append(outputs.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init: float = config.TEMPERATURE_INIT,
    lr: float = config.TEMPERATURE_LR,
    num_steps: int = config.TEMPERATURE_NUM_STEPS,
) -> float:
    """Learn a single scalar temperature T that minimizes cross-entropy
    loss on (logits / T) vs the true labels. Only T is learned — the
    model's weights are never touched.
    """
    log_temperature = nn.Parameter(torch.log(torch.tensor(init)))
    optimizer = torch.optim.Adam([log_temperature], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)  # keeps T positive during optimization
        loss = criterion(logits / temperature, labels)
        loss.backward()
        optimizer.step()

    return torch.exp(log_temperature).item()


def confidences_and_correctness(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float
) -> tuple[list[float], list[bool]]:
    """Apply temperature scaling, then return (max-softmax confidence,
    is-prediction-correct) per example — the inputs compute_ece() expects.
    """
    probs = torch.softmax(logits / temperature, dim=1)
    confidences, preds = torch.max(probs, dim=1)
    correct = preds == labels

    return confidences.tolist(), correct.tolist()


def ece_at_temperature(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float, num_bins: int
) -> float:
    confidences, correct = confidences_and_correctness(logits, labels, temperature)
    return compute_ece(confidences, correct, num_bins)


def reliability_diagram_data(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float, num_bins: int
) -> tuple[list[float], list[float]]:
    """Returns (bin_confidences, bin_accuracies) — one value per non-empty
    bin — for plotting a reliability diagram. Bin i covers confidence
    range (i/num_bins, (i+1)/num_bins].
    """
    confidences, correct = confidences_and_correctness(logits, labels, temperature)
    bin_edges = [i / num_bins for i in range(num_bins + 1)]

    bin_confidences = []
    bin_accuracies = []

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            (conf, is_correct)
            for conf, is_correct in zip(confidences, correct)
            if (lo < conf <= hi) or (i == 0 and conf == lo)
        ]
        if not in_bin:
            continue
        bin_confidences.append(sum(c for c, _ in in_bin) / len(in_bin))
        bin_accuracies.append(sum(is_correct for _, is_correct in in_bin) / len(in_bin))

    return bin_confidences, bin_accuracies
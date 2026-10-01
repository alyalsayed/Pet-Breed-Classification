from .metrics import compute_top1_accuracy, compute_macro_f1, compute_ece

# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_confidences, all_correct = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_correct.extend((preds == labels).cpu().tolist())

    return {
        "val_loss": total_loss / len(loader.dataset),
        "val_top1": compute_top1_accuracy(all_preds, all_labels),
        "val_macro_f1": compute_macro_f1(all_preds, all_labels, config.NUM_CLASSES),
        "val_ece": compute_ece(all_confidences, all_correct, config.ECE_NUM_BINS),
    }

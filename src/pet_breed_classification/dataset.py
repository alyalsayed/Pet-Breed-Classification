
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config, features
from .manifest import load_split_records


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class PetBreedDataset(Dataset):
    """Wraps a list of manifest records (already filtered to one split)."""

    def __init__(self, records: list[dict], transform: transforms.Compose):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = features.load_image(record["path"])
        image = self.transform(image)
        label = record["class_index"]
        return image, label


def build_train_transform() -> transforms.Compose:
    """Same resize/crop/normalize as features.get_eval_transform(), with one
    light augmentation (random horizontal flip) added on top. Augmentation
    stays here, in training code only — never in features.py, and never
    applied to validation/test/serving.
    """
    return transforms.Compose(
        [
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_records = load_split_records("train")
    val_records = load_split_records("val")

    train_ds = PetBreedDataset(train_records, build_train_transform())
    val_ds = PetBreedDataset(val_records, features.get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


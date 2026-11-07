"""Dataset loading, preprocessing, and reproducible data loaders."""

from __future__ import annotations

from collections import Counter

import torch
from src.cache_config import configure_huggingface_cache

configure_huggingface_cache()

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTImageProcessor

DATASET_ID = "funkepal/medicinal_plant_images"
DEFAULT_MODEL_ID = "google/vit-base-patch16-224-in21k"


class PlantImageDataset(Dataset):
    def __init__(self, rows, transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        item = self.rows[index]
        image = item["image"].convert("RGB")
        return self.transform(image), int(item["label"])


def load_splits(validation_fraction: float = 0.15, seed: int = 42):
    dataset = load_dataset(DATASET_ID)
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Dataset must provide named 'train' and 'test' splits.")

    train_rows = dataset["train"]
    test_rows = dataset["test"]
    label_feature = train_rows.features["label"]
    class_names = list(label_feature.names)
    if not class_names:
        raise ValueError("Dataset label feature does not define class names.")

    indices = list(range(len(train_rows)))
    labels = [int(train_rows[i]["label"]) for i in indices]
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=labels,
    )
    return (
        train_rows.select(train_indices),
        train_rows.select(validation_indices),
        test_rows,
        class_names,
    )


def build_loaders(train_rows, validation_rows, test_rows, processor, batch_size: int):
    mean, std = processor.image_mean, processor.image_std
    common = [
        transforms.Resize((processor.size["height"], processor.size["width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), *common]
    )
    eval_transform = transforms.Compose(common)

    def make(rows, transform, shuffle):
        return DataLoader(
            PlantImageDataset(rows, transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        make(train_rows, train_transform, True),
        make(validation_rows, eval_transform, False),
        make(test_rows, eval_transform, False),
    )


def class_weights(rows, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    weights = [len(rows) / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32, device=device)

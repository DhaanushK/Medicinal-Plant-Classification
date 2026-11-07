"""Load a trained checkpoint and classify one image."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from src.model import HybridPlantClassifier


def load_predictor(checkpoint_path: str | Path, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = HybridPlantClassifier(
        len(checkpoint["class_names"]),
        checkpoint["model_id"],
        vit_config=checkpoint["vit_config"],
        load_pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["image_mean"], checkpoint["image_std"]),
        ]
    )
    return model, transform, checkpoint["class_names"], device


def predict_image(image: Image.Image, model, transform, class_names, device, top_k: int = 3):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(class_names)))
    return [(class_names[i], float(score)) for score, i in zip(values, indices)]

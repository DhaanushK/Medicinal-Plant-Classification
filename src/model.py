"""Hybrid CNN and ViT classifier with a serializable checkpoint format."""

from __future__ import annotations

import torch
from torch import nn
from transformers import ViTConfig, ViTForImageClassification


class HybridPlantClassifier(nn.Module):
    """Combine pretrained ViT and CNN logits through a small fusion head."""

    def __init__(
        self,
        num_labels: int,
        pretrained_model: str,
        vit_config: dict | None = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if load_pretrained:
            self.vit = ViTForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
            )
        else:
            config = ViTConfig.from_dict(vit_config)
            config.num_labels = num_labels
            self.vit = ViTForImageClassification(config)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, num_labels),
        )
        self.fusion = nn.Sequential(
            nn.Linear(num_labels * 2, num_labels),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_logits = self.vit(pixel_values=pixel_values).logits
        cnn_logits = self.cnn(pixel_values)
        return self.fusion(torch.cat((vit_logits, cnn_logits), dim=1))

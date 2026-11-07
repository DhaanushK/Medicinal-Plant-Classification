"""Train, validate, checkpoint, and finally evaluate the hybrid classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from src.cache_config import configure_huggingface_cache

configure_huggingface_cache()

from transformers import ViTImageProcessor

from src.data import DEFAULT_MODEL_ID, build_loaders, class_weights, load_splits
from src.model import HybridPlantClassifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, collect_predictions=False):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    targets, predictions = [], []
    amp_enabled = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp_enabled
        ):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        count = labels.size(0)
        total_loss += loss.item() * count
        correct += (logits.argmax(1) == labels).sum().item()
        seen += count
        if collect_predictions:
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    metrics = {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}
    return (metrics, targets, predictions) if collect_predictions else metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_rows, validation_rows, test_rows, class_names = load_splits(
        args.validation_fraction, args.seed
    )
    processor = ViTImageProcessor.from_pretrained(args.model_id)
    train_loader, validation_loader, test_loader = build_loaders(
        train_rows, validation_rows, test_rows, processor, args.batch_size
    )

    model = HybridPlantClassifier(len(class_names), args.model_id).to(device)
    weights = class_weights(train_rows, len(class_names), device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "medicinal_plant_hybrid.pt"
    best_validation_accuracy, stale_epochs = -1.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        validation_metrics = run_epoch(model, validation_loader, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        print(json.dumps(record))

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "model_id": args.model_id,
                    "image_mean": processor.image_mean,
                    "image_std": processor.image_std,
                    "image_size": processor.size["height"],
                    "seed": args.seed,
                    "vit_config": model.vit.config.to_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping: validation accuracy did not improve.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    test_metrics, targets, predictions = run_epoch(
        model, test_loader, criterion, device, collect_predictions=True
    )

    results = {
        "dataset": "funkepal/medicinal_plant_images",
        "model": args.model_id,
        "seed": args.seed,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "test_count": len(test_rows),
        "class_names": class_names,
        "history": history,
        "test": test_metrics,
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(len(class_names)))
        ).tolist(),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Final held-out test metrics: {json.dumps(test_metrics)}")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()

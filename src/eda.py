"""Summarize the dataset and save reproducible exploratory plots."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.cache_config import configure_huggingface_cache

configure_huggingface_cache()

from datasets import load_dataset

from src.data import DATASET_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_images < 1:
        parser.error("--max-images must be positive")

    dataset = load_dataset(DATASET_ID)
    names = list(dataset["train"].features["label"].names)
    train_counts = Counter(int(row) for row in dataset["train"]["label"])
    test_counts = Counter(int(row) for row in dataset["test"]["label"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(ids, [train_counts[i] for i in ids], color="#3b8069")
    ax.set(title="Training images by class", xlabel="Plant class", ylabel="Image count")
    ax.set_xticks(ids, names, rotation=70, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    width = 0.42
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(ids - width / 2, [train_counts[i] for i in ids], width, label="Train")
    ax.bar(ids + width / 2, [test_counts[i] for i in ids], width, label="Test")
    ax.set(title="Training and test class counts", xlabel="Plant class", ylabel="Image count")
    ax.set_xticks(ids, names, rotation=70, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "train_test_distribution.png", dpi=160)
    plt.close(fig)

    rng = random.Random(args.seed)
    examples = []
    classes_seen = set()
    for i in rng.sample(range(len(dataset["train"])), min(len(dataset["train"]), args.max_images)):
        row = dataset["train"][i]
        label = int(row["label"])
        if label not in classes_seen:
            examples.append((row["image"].convert("RGB"), names[label]))
            classes_seen.add(label)
    columns = 5
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3 * rows), squeeze=False)
    for ax, (image, label) in zip(axes.flat, examples):
        ax.imshow(image)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in axes.flat[len(examples) :]:
        ax.axis("off")
    fig.suptitle("One example image per observed class")
    fig.tight_layout()
    fig.savefig(args.output_dir / "class_examples.png", dpi=160)
    plt.close(fig)

    sampled_rows = rng.sample(range(len(dataset["train"])), min(len(dataset["train"]), args.max_images))
    sizes, channel_means = [], []
    for i in sampled_rows:
        image = dataset["train"][i]["image"].convert("RGB")
        sizes.append(image.size)
        channel_means.append(np.asarray(image, dtype=np.float32).mean(axis=(0, 1)))
    sizes = np.asarray(sizes)
    channel_means = np.asarray(channel_means)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sizes[:, 0], sizes[:, 1], alpha=0.55, s=14)
    ax.set(title="Image dimensions in a reproducible sample", xlabel="Width (pixels)", ylabel="Height (pixels)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "image_dimensions.png", dpi=160)
    plt.close(fig)

    correlation = np.corrcoef(channel_means.T)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set(title="Correlation of per-image RGB means", xticks=range(3), yticks=range(3),
           xticklabels=["R", "G", "B"], yticklabels=["R", "G", "B"])
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(args.output_dir / "rgb_mean_correlation.png", dpi=160)
    plt.close(fig)

    summary = {
        "dataset": DATASET_ID,
        "train_count": len(dataset["train"]),
        "test_count": len(dataset["test"]),
        "class_count": len(names),
        "class_names": names,
        "train_counts_by_class": {names[i]: train_counts[i] for i in ids},
        "test_counts_by_class": {names[i]: test_counts[i] for i in ids},
        "image_size_sample_count": int(len(sampled_rows)),
        "image_width_range": [int(sizes[:, 0].min()), int(sizes[:, 0].max())],
        "image_height_range": [int(sizes[:, 1].min()), int(sizes[:, 1].max())],
        "mean_rgb_sample": channel_means.mean(axis=0).round(2).tolist(),
        "seed": args.seed,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_count", "test_count", "class_count", "image_size_sample_count")}, indent=2))
    print(f"EDA outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()

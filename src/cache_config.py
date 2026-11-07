"""Configure optional Hugging Face cache locations before importing HF libraries."""

from __future__ import annotations

import os
from pathlib import Path


def configure_huggingface_cache() -> None:
    cache_root = os.environ.get("MEDICINAL_PLANT_CACHE")
    if not cache_root:
        return
    root = Path(cache_root)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["HF_XET_CACHE"] = str(root / "xet")


configure_huggingface_cache()

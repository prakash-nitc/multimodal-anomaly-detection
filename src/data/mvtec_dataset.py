"""
MVTec AD Dataset Loader
=======================
PyTorch Dataset for the MVTec Anomaly Detection dataset.
Loads test images with binary labels (0=good, 1=anomalous) for evaluation.

Dataset structure expected:
    mvtec_anomaly_detection/
    ├── bottle/
    │   ├── test/
    │   │   ├── good/
    │   │   ├── broken_large/
    │   │   └── ...
    │   └── train/
    │       └── good/
    ├── cable/
    └── ...
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

from PIL import Image
from torch.utils.data import Dataset


# All 15 MVTec AD categories
MVTEC_CATEGORIES = [
    # Textures (5)
    "carpet", "grid", "leather", "tile", "wood",
    # Objects (10)
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]


class MVTecDataset(Dataset):
    """MVTec AD test set loader for anomaly detection evaluation.

    Args:
        root_dir: Path to the root of MVTec AD dataset (e.g., 'data/mvtec_anomaly_detection').
        category: One of the 15 MVTec AD categories (e.g., 'bottle').
        split: 'test' or 'train'. Default is 'test'.
        transform: Optional torchvision transform to apply to images.
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "test",
        transform: Optional[Callable] = None,
    ):
        assert category in MVTEC_CATEGORIES, (
            f"Unknown category '{category}'. Must be one of: {MVTEC_CATEGORIES}"
        )
        assert split in ("test", "train"), f"Split must be 'test' or 'train', got '{split}'"

        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []  # (image_path, label)
        self._load_samples()

    def _load_samples(self):
        """Scan the directory and collect (image_path, label) pairs."""
        split_dir = self.root_dir / self.category / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"MVTec AD directory not found: {split_dir}\n"
                f"Please download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad"
            )

        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            if not defect_dir.is_dir():
                continue

            # 'good' folder = normal (label 0), everything else = anomalous (label 1)
            label = 0 if defect_type == "good" else 1

            for img_file in sorted(os.listdir(defect_dir)):
                if img_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self.samples.append((str(defect_dir / img_file), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, self.category, img_path

    @property
    def num_normal(self) -> int:
        return sum(1 for _, label in self.samples if label == 0)

    @property
    def num_anomalous(self) -> int:
        return sum(1 for _, label in self.samples if label == 1)

    def __repr__(self) -> str:
        return (
            f"MVTecDataset(category='{self.category}', split='{self.split}', "
            f"total={len(self)}, normal={self.num_normal}, anomalous={self.num_anomalous})"
        )

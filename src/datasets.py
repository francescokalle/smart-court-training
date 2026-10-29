"""Dataset classes for tennis training."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TrackNetDataset(Dataset):
    """TrackNet dataset: 3 consecutive frames → heatmap target.

    Expects directory structure:
        root/
          images/gameX/ClipY/0000.png ...
          gts/gameX/ClipY/0000.png ...
          labels_train.csv
    """

    def __init__(self, root: str, split: str = "train", img_size: Tuple[int, int] = (288, 512)):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.samples = self._load_samples()

    def _load_samples(self) -> List[dict]:
        # Stub: parse CSV labels and build sample list
        samples = []
        csv_path = self.root / f"labels_{self.split}.csv"
        if not csv_path.exists():
            return samples
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                samples.append({
                    "frame": int(r["Frame"]),
                    "vis": int(r["Visibility"]),
                    "x": float(r["X"]),
                    "y": float(r["Y"]),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Stub: load 3 frames, resize, normalize, generate heatmap
        # Real implementation: load frames from images/, create input tensor (3, H, W)
        # and target heatmap (H, W) with gaussian at (x, y)
        dummy_input = torch.zeros(3, *self.img_size, dtype=torch.float32)
        dummy_target = torch.zeros(*self.img_size, dtype=torch.float32)
        return dummy_input, dummy_target


class CourtKeypointsDataset(Dataset):
    """Court keypoints dataset for YOLO-pose.

    Expects YOLO-pose format:
        images/train/*.jpg
        labels/train/*.txt  (class x y w h kp1_x kp1_y kp1_vis ...)
    """

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.samples = list(self.img_dir.glob("*.jpg")) if self.img_dir.exists() else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Stub: return image and label path; real impl parses YOLO-pose txt
        return img, str(label_path)

"""Fine-tune YOLOv8 for tennis (player/racket/ball)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(data_yaml: str, model: str = "yolov8n.pt", epochs: int = 100, imgsz: int = 640,
          batch: int = 16, device: str = "0", project: str = "runs", name: str = "tennis"):
    model = YOLO(model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        verbose=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="tennis")
    args = parser.parse_args()
    train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, name=args.name)


if __name__ == "__main__":
    main()

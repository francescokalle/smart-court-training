"""Export trained models to ONNX for Jetson deployment."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def export_yolo(model_path: str, imgsz: int = 640, out_dir: str = "exports"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    print(f"YOLO ONNX exported: {path}")
    return path


def export_tracknet(model_path: str, img_size: tuple = (288, 512), out_dir: str = "exports"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Load model state dict — adjust architecture to match training
    import torch.nn as nn
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 3, *img_size)
    onnx_path = out / "tracknet.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["heatmap"],
        dynamic_axes={"input": {0: "batch"}, "heatmap": {0: "batch"}},
        opset_version=12,
    )
    print(f"TrackNet ONNX exported: {onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--type", choices=["yolo", "tracknet"], required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out_dir", default="exports")
    args = parser.parse_args()
    if args.type == "yolo":
        export_yolo(args.model, args.imgsz, args.out_dir)
    else:
        export_tracknet(args.model, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

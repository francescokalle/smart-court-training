"""Preprocess TrackNet dataset into training format.

Run on Colab after mounting Drive:
    python scripts/prepare_tracknet.py --data_root /content/drive/MyDrive/smart-court-data/tracknet
"""
import argparse
import csv
import os
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, fps: int = 30):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{idx:04d}.png"), frame)
        idx += 1
    cap.release()
    return idx


def parse_labels(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "frame": int(r["Frame"]),
                "vis": int(r["Visibility"]),
                "x": float(r["X"]),
                "y": float(r["Y"]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    print(f"Data root: {root}")
    # TODO: implement full preprocessing per TrackNet format
    # - extract frames from videos
    # - generate 3-frame input stacks
    # - generate heatmap targets
    # - save as .npy or .pt
    print("Preprocessing stub — implement per TrackNetV3 preprocess.py")


if __name__ == "__main__":
    main()

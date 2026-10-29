"""Validazione 'a secco' su video YouTube/MP4 (senza hardware fisico).

Esegui sul Jetson o su PC GPU:
    python scripts/validate.py --video test.mp4 --yolo_engine yolov8n_tennis.engine
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo_engine", default=None)
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    frame_count = 0
    while frame_count < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        # TODO: run detection + tracking + metrics
        frame_count += 1

    cap.release()
    print(f"Processed {frame_count} frames")
    print("Validation stub — implement detection + tracking + metrics")


if __name__ == "__main__":
    main()

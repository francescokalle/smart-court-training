"""Convert court keypoints dataset to YOLO-pose format.

Run on Colab:
    python scripts/prepare_court.py --zip_path /content/drive/MyDrive/smart-court-data/court_keypoints/tennis_court_det_dataset.zip --out_dir /content/court_yolo
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    print(f"Zip: {args.zip_path}")
    print(f"Out: {args.out_dir}")
    # TODO: unzip, convert annotations to YOLO-pose format
    # - images → train/val
    # - labels → YOLO-pose txt (class + 14 keypoints)
    print("Preprocessing stub — convert to YOLO-pose format")


if __name__ == "__main__":
    main()

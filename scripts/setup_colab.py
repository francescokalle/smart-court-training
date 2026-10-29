#!/usr/bin/env python3
"""Setup completo su Colab con un solo comando."""
import argparse, os, zipfile, subprocess, sys

def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    # 1. Installa dipendenze
    print("📦 Installazione dipendenze...")
    run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "ultralytics",
         "torch", "torchvision", "opencv-python", "numpy", "tqdm"])

    # 2. Scarica dataset da HF
    print("⬇️  Download dataset da Hugging Face...")
    from huggingface_hub import snapshot_download
    d = snapshot_download("realKalle/smart-court-datasets", repo_type="dataset",
                          local_dir="/content/data", token=args.token)
    print(f"✅ Dataset in {d}")

    # 3. Estrai zip
    for z in ["/content/data/tracknet/Dataset.zip",
              "/content/data/court_keypoints/tennis_court_det_dataset.zip"]:
        if os.path.exists(z):
            out = os.path.dirname(z) + "/extracted"
            os.makedirs(out, exist_ok=True)
            print(f"📂 Estrazione {z}...")
            with zipfile.ZipFile(z) as f:
                f.extractall(out)
            print(f"✅ Estratto in {out}")

    print("\n🎉 Setup completato!")
    print("   Dati in: /content/data")
    print("   YOLO:    python src/train_yolo.py --data <data.yaml> --epochs 100")
    print("   TrackNet: vedi notebooks/02_train_tracknet.ipynb")

if __name__ == "__main__":
    main()
# setup

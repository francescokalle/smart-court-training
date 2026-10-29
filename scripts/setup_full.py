#!/usr/bin/env python3
"""Setup completo su Colab: dipendenze, dataset, modelli, training.

Esegue in sequenza:
  1. Installazione dipendenze
  2. Download dataset da Hugging Face (TrackNet + Court Keypoints)
  3. Estrazione zip
  4. Download modelli pretrained (CourtSide, Court Keypoints)
  5. (Opzionale) Download dataset YOLO da Roboflow + training

Uso:
    python setup_full.py --hf-token TOKEN [--roboflow-key KEY] [--epochs 100] [--skip-training]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run(cmd, check=True, capture=False, timeout=None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    try:
        if capture:
            r = subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
            if r.stdout:
                print(r.stdout)
            if r.stderr:
                print(r.stderr, file=sys.stderr)
            return r
        else:
            return subprocess.run(cmd, check=check, timeout=timeout)
    except subprocess.CalledProcessError as e:
        log(f"Comando fallito (exit {e.returncode})", "ERROR")
        raise
    except subprocess.TimeoutExpired:
        log("Timeout scaduto", "ERROR")
        raise


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def download_hf_datasets(hf_token):
    log("Download dataset da Hugging Face...")
    from huggingface_hub import snapshot_download
    d = snapshot_download(
        "realKalle/smart-court-datasets",
        repo_type="dataset",
        local_dir="/content/data",
        token=hf_token,
    )
    log(f"Dataset scaricati in {d}")
    return d


def extract_zips():
    log("Estrazione zip...")
    zips = [
        "/content/data/tracknet/Dataset.zip",
        "/content/data/court_keypoints/tennis_court_det_dataset.zip",
    ]
    for z in zips:
        if not os.path.exists(z):
            log(f"  {z} non trovato, skip.", "WARN")
            continue
        out = os.path.dirname(z) + "/extracted"
        ensure_dir(out)
        log(f"  Estrazione {z} → {out}")
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        log(f"  ✅ {z} estratto")
    return True


def download_pretrained_models(hf_token):
    log("Download modelli pretrained...")
    from huggingface_hub import hf_hub_download
    models = [
        ("models/courtside/model.pt", "/content/models/courtside/model.pt"),
        ("models/court_keypoints/keypoints_model_geoaug.pth", "/content/models/court_keypoints/keypoints_model_geoaug.pth"),
    ]
    for remote, local in models:
        ensure_dir(os.path.dirname(local))
        hf_hub_download(
            "realKalle/smart-court-datasets",
            remote,
            repo_type="dataset",
            local_dir="/content",
            token=hf_token,
        )
        # hf_hub_download salva in /content/<remote>, spostiamo se necessario
        src = os.path.join("/content", remote)
        if os.path.exists(src) and src != local:
            shutil.move(src, local)
        log(f"  ✅ {remote} → {local}")
    return True


def download_roboflow_dataset(api_key):
    log("Download dataset YOLO da Roboflow...")
    try:
        from roboflow import Roboflow
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "roboflow"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("abiya-thesis").project("tennis-player-and-tennis-ball-ei0vm")
    version = project.version(6)
    dataset = version.download("yolov8")
    log(f"  Dataset scaricato in: {dataset.location}")
    return dataset.location


def find_data_yaml(root="/content"):
    for dirpath, _, filenames in os.walk(root):
        if "data.yaml" in filenames:
            return os.path.join(dirpath, "data.yaml")
    return None


def train_yolo(data_yaml, epochs=100, imgsz=640, batch=16):
    log("Avvio training YOLO...")
    train_script = "/content/smart-court-training/src/train_yolo.py"
    if not os.path.exists(train_script):
        # clona il repo training se non presente
        log("Clone repo training...")
        run(["git", "clone", "https://github.com/francescokalle/smart-court-training.git",
             "/content/smart-court-training"])
    cmd = [
        sys.executable, train_script,
        "--data", data_yaml,
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "0",
    ]
    run(cmd)
    log("✅ Training completato!")


def main():
    parser = argparse.ArgumentParser(description="Setup completo Colab")
    parser.add_argument("--hf-token", required=True, help="Token Hugging Face")
    parser.add_argument("--roboflow-key", default=None, help="API key Roboflow (opzionale)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs training YOLO")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--skip-training", action="store_true", help="Salta il training")
    parser.add_argument("--skip-roboflow", action="store_true", help="Salta download Roboflow")
    args = parser.parse_args()

    print("=" * 60)
    print("  SETUP COMPLETO COLAB — SMART COURT TENNIS")
    print("=" * 60)

    # 1. Dipendenze
    log("Fase 1/6: Installazione dipendenze...")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub", "ultralytics", "torch", "torchvision",
         "opencv-python", "numpy", "tqdm", "roboflow", "pyyaml"])
    log("✅ Dipendenze installate")

    # 2. Download HF datasets
    log("Fase 2/6: Download dataset da Hugging Face...")
    try:
        download_hf_datasets(args.hf_token)
    except Exception as e:
        log(f"Errore download HF: {e}", "ERROR")
        sys.exit(1)

    # 3. Estrazione
    log("Fase 3/6: Estrazione zip...")
    try:
        extract_zips()
    except Exception as e:
        log(f"Errore estrazione: {e}", "ERROR")
        sys.exit(1)

    # 4. Modelli pretrained
    log("Fase 4/6: Download modelli pretrained...")
    try:
        download_pretrained_models(args.hf_token)
    except Exception as e:
        log(f"Errore download modelli: {e}", "ERROR")

    # 5. Roboflow dataset (opzionale)
    data_yaml = None
    if args.roboflow_key and not args.skip_roboflow:
        log("Fase 5/6: Download dataset YOLO da Roboflow...")
        try:
            location = download_roboflow_dataset(args.roboflow_key)
            data_yaml = find_data_yaml(location)
            if data_yaml:
                log(f"  data.yaml trovato: {data_yaml}")
            else:
                log("  data.yaml non trovato, training saltato.", "WARN")
        except Exception as e:
            log(f"Errore Roboflow: {e}", "ERROR")
    else:
        log("Fase 5/6: Skip Roboflow (nessuna API key fornita).")
        # cerca data.yaml esistente
        data_yaml = find_data_yaml("/content")

    # 6. Training
    if not args.skip_training and data_yaml:
        log("Fase 6/6: Training YOLO...")
        try:
            train_yolo(data_yaml, args.epochs, args.imgsz, args.batch)
        except Exception as e:
            log(f"Errore training: {e}", "ERROR")
            log("Puoi riprovare manualmente con:")
            log(f"  !python /content/smart-court-training/src/train_yolo.py --data {data_yaml} --epochs {args.epochs}")
    elif not data_yaml:
        log("Fase 6/6: Nessun data.yaml trovato, training saltato.")
        log("Per fare training, esporta un dataset da Roboflow e riesegui con --roboflow-key KEY")

    print("\n" + "=" * 60)
    print("  ✅ SETUP COMPLETATO")
    print("=" * 60)
    print("  Dataset HF:     /content/data")
    print("  Modelli:        /content/models")
    print("  Training code:  /content/smart-court-training")
    if data_yaml:
        print(f"  data.yaml:      {data_yaml}")
    print("=" * 60)


if __name__ == "__main__":
    main()

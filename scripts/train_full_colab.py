#!/usr/bin/env python3
"""Training completo su Colab: merge dataset, training, export, upload.

Esegue tutto in un colpo:
  1. Installa dipendenze
  2. Scarica i 3 dataset Roboflow
  3. Merge in un unico dataset YOLO (person + tennis_ball)
  4. Training YOLOv8s (200 epochs, early stopping)
  5. Export ONNX
  6. Upload modelli su Hugging Face
  7. Stampa link download

Uso:
    python train_full_colab.py --hf-token TOKEN --roboflow-key KEY [--epochs 200] [--model yolov8s.pt]
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def run(cmd, check=True):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=check)


def install_deps():
    log("Installazione dipendenze...")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "roboflow", "huggingface_hub", "pyyaml", "opencv-python", "numpy", "tqdm"])
    log("✅ Dipendenze installate")


def download_roboflow(api_key):
    log("Download dataset Roboflow...")
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    specs = [
        ("abiya-thesis", "tennis-player-and-tennis-ball-ei0vm", 6),
        ("tennis-ball-tracking-kte4k", "tennis-ball-tracking-1wnxz", 2),
        ("tennis-ai", "tennis-player-detection-4vnfq", 3),
    ]
    locations = []
    for ws, proj, ver in specs:
        try:
            project = rf.workspace(ws).project(proj)
            dataset = project.version(ver).download("yolov8")
            locations.append(Path(dataset.location))
            log(f"  ✅ {ws}/{proj} -> {dataset.location}")
        except Exception as e:
            log(f"  ⚠️  {ws}/{proj}: {e}", "WARN")
    if not locations:
        raise RuntimeError("Nessun dataset Roboflow scaricato")
    return locations


def load_names(data_yaml):
    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n).lower() for n in names]


def map_class(name):
    """Mappa nome classe -> 0=person, 1=tennis_ball. None = ignora."""
    n = name.lower()
    if any(k in n for k in ("ball", "palla", "tennis ball")):
        return 1
    if any(k in n for k in ("player", "person", "giocatore", "tennis player")):
        return 0
    return None


def collect_split(dataset_dir, split):
    """Raccogli (image, label) per uno split."""
    img_dir = dataset_dir / split / "images"
    lbl_dir = dataset_dir / split / "labels"
    if not img_dir.exists():
        # fallback: Roboflow a volte usa 'valid' invece di 'val'
        alt = {"valid": "val", "val": "valid"}
        if split in alt:
            img_dir = dataset_dir / alt[split] / "images"
            lbl_dir = dataset_dir / alt[split] / "labels"
    if not img_dir.exists():
        return []
    pairs = []
    for img in img_dir.glob("*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        pairs.append((img, lbl if lbl.exists() else None))
    return pairs


def merge_datasets(locations, out_dir):
    log("Merge dataset...")
    out = Path(out_dir)
    for sub in ("images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    all_pairs = []  # (img, lbl, names_map)
    for ds in locations:
        data_yaml = ds / "data.yaml"
        if not data_yaml.exists():
            log(f"  ⚠️  data.yaml mancante in {ds}", "WARN")
            continue
        names = load_names(data_yaml)
        cmap = {i: map_class(n) for i, n in enumerate(names)}
        log(f"  {ds.name}: classi={names} -> mapping={cmap}")
        for split in ("train", "valid", "test"):
            for img, lbl in collect_split(ds, split):
                all_pairs.append((img, lbl, cmap, ds.name))

    if not all_pairs:
        raise RuntimeError("Nessuna immagine trovata nei dataset")

    random.seed(42)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": all_pairs[:n_train],
        "val": all_pairs[n_train:n_train + n_val],
        "test": all_pairs[n_train + n_val:],
    }

    counts = {}
    for split, pairs in splits.items():
        counts[split] = {"images": 0, "labels": 0, "background": 0}
        for idx, (img, lbl, cmap, dsname) in enumerate(pairs):
            stem = f"{dsname}_{img.stem}_{idx:06d}"
            dst_img = out / "images" / split / (stem + img.suffix.lower())
            shutil.copy2(img, dst_img)
            counts[split]["images"] += 1

            dst_lbl = out / "labels" / split / (stem + ".txt")
            lines_out = []
            if lbl is not None and lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls = int(float(parts[0]))
                    new_cls = cmap.get(old_cls)
                    if new_cls is None:
                        continue
                    lines_out.append(" ".join([str(new_cls)] + parts[1:]))
            if lines_out:
                dst_lbl.write_text("\n".join(lines_out) + "\n")
                counts[split]["labels"] += len(lines_out)
            else:
                dst_lbl.write_text("")
                counts[split]["background"] += 1

    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person", 1: "tennis_ball"},
    }
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    log(f"✅ Merge completato: {counts}")
    return out / "data.yaml", counts


def train_yolo(data_yaml, model_name, epochs, imgsz, batch, device, project):
    log("Training YOLO...")
    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        project=project,
        name="tennis_full",
        exist_ok=True,
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    log(f"✅ Training completato: {best}")
    return best, results


def export_onnx(best_pt, imgsz):
    log("Export ONNX...")
    from ultralytics import YOLO
    model = YOLO(str(best_pt))
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    log(f"✅ ONNX: {onnx_path}")
    return Path(onnx_path)


def upload_to_hf(hf_token, files, repo_id, prefix):
    log("Upload su Hugging Face...")
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    urls = []
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        path_in_repo = f"{prefix}/{f.name}"
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
        urls.append(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}")
        log(f"  ✅ {f.name} -> {path_in_repo}")
    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True)
    parser.add_argument("--roboflow-key", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--hf-repo", default="realKalle/smart-court-datasets")
    args = parser.parse_args()

    print("=" * 62)
    print("  TRAINING COMPLETO — SMART COURT TENNIS")
    print("=" * 62)

    install_deps()

    # Verifica GPU
    try:
        import torch
        if not torch.cuda.is_available():
            log("⚠️  GPU non disponibile! Il training su CPU sarà lentissimo.", "WARN")
            args.device = "cpu"
        else:
            log(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    locations = download_roboflow(args.roboflow_key)
    data_yaml, counts = merge_datasets(locations, "/content/merged_dataset")
    best_pt, results = train_yolo(data_yaml, args.model, args.epochs, args.imgsz,
                                  args.batch, args.device, "/content/runs")
    onnx_path = export_onnx(best_pt, args.imgsz)

    # Report
    report = {
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_done": getattr(results, "epoch", args.epochs),
        "dataset_counts": counts,
        "classes": {"0": "person", "1": "tennis_ball"},
        "best_pt": str(best_pt),
        "onnx": str(onnx_path),
    }
    report_path = Path("/content/training_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    urls = upload_to_hf(args.hf_token, [best_pt, onnx_path, report_path],
                        args.hf_repo, "models/trained")

    print("\n" + "=" * 62)
    print("  ✅ TUTTO COMPLETATO")
    print("=" * 62)
    print(f"  Modello .pt:  {best_pt}")
    print(f"  Modello .onnx: {onnx_path}")
    print("\n  Link download:")
    for u in urls:
        print(f"   {u}")
    print("=" * 62)


if __name__ == "__main__":
    main()

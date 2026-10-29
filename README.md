# Smart Court Tennis — Training Pipeline (Cloud GPU)

Questa cartella contiene tutto il necessario per **allenare, esportare e ottimizzare** i modelli AI su GPU cloud (Colab Pro / RunPod / AWS).  
**Non eseguire training su macchine senza GPU**: è lento e inutile.

## Modelli da allenare

| Modello | Architettura | Dataset | Output |
|---|---|---|---|
| Player/Racket/Ball Detection | YOLOv8n fine-tune | CourtSide + Roboflow Tennis Player/Ball | `yolov8n_tennis.pt` → ONNX → TensorRT |
| Court Keypoints | YOLOv8-pose o ResNet-50 | Tennis Court Keypoints (8.841 img) | `court_keypoints.pt` → ONNX → TensorRT |
| Ball Tracking | TrackNet (3-frame heatmap) | TrackNet Tennis (broadcast) | `tracknet.pt` → ONNX → TensorRT |

## Struttura

```
smart-court-training/
├── configs/          # YAML di training
├── src/              # Moduli Python riusabili
├── scripts/          # Preprocessing dataset
├── notebooks/        # Jupyter per Colab
└── docs/             # Guide step-by-step
```

## Quick Start (Colab Pro)

1. Apri `notebooks/01_train_yolo.ipynb` in Colab.
2. Seleziona GPU (Runtime → Change runtime type → T4/A100).
3. Monta Google Drive con i dataset (vedi `docs/DATASET_SETUP.md`).
4. Esegui le celle in ordine.

## Cosa devi fare tu (checklist)

- [ ] Creare account Google Colab Pro (o RunPod/AWS).
- [ ] Caricare i dataset su Google Drive (vedi docs).
- [ ] Eseguire i notebook in ordine.
- [ ] Scaricare i modelli `.pt` / `.onnx` / `.engine` sul Jetson.

## Flusso completo

```
Dataset (Drive) → Training (Colab GPU) → Export ONNX → TensorRT (Jetson) → Validazione (video YouTube)
```

# Smart Court Tennis Analysis — TUI v3

## Cosa include (implementato)
- ✅ YOLO detection (player + ball)
- ✅ ByteTrack (ID persistenti giocatori)
- ✅ CourtROI (filtro geometrico campo, elimina spettatori)
- ✅ BallKalman (filtro Kalman palla, elimina falsi positivi)
- ✅ Bounce detection (inversione velocità verticale)
- ✅ Report JSON (rally, ball, bounce, player tracks)
- ✅ Heatmap giocatori
- ✅ Highlights FFmpeg

## Requisiti
- Python 3.10+
- ffmpeg nel PATH
- Modello YOLO `best.pt` (o `model.pt`)

## Setup Windows
1. Installa Python da https://python.org (spunta "Add to PATH")
2. Installa dipendenze:
   ```
   pip install ultralytics opencv-python numpy torch torchvision
   ```
3. Installa ffmpeg: https://ffmpeg.org/download.html → aggiungi `bin` al PATH
4. Metti nella stessa cartella:
   - `smartcourt_tui_v3.py`
   - `smartcourt_roi_calibrate.py`
   - `postprocessing.py`
   - `best.pt` (modello YOLO)

## Uso

### 1. Calibra il campo (una volta per video)
```
python smartcourt_roi_calibrate.py video.mp4
```
Clicca i 4 angoli del campo in ordine: alto-sx, alto-dx, basso-dx, basso-sx.
Salva `roi.json`.

### 2. Avvia il TUI
```
python smartcourt_tui_v3.py
```
- Opzione 1: analizza video (usa `roi.json` se presente)
- Opzione 2: ricalibra ROI
- Opzione 3: info modello

## Output
- `output/report.json` — statistiche
- `output/heatmap_players.jpg` — heatmap giocatori
- `output/highlights.mp4` — highlights rally
- `output/frames/` — frame annotati

## Cosa NON è ancora implementato
- Court Keypoints automatico (modello ResNet-50) — richiede integrazione specifica
- TrackNet ball tracking (modello U-Net 3-frame) — richiede preprocessing
- Omografia in metri reali — richiede calibrazione 4 punti + H matrix
- TensorRT — solo su Jetson

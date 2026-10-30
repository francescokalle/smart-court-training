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

**Se un angolo è tagliato dal bordo del video:**
- usa la **rotella** per dezoomare (il video rimpicciolisce e appare lo spazio grigio attorno)
- clicca direttamente nello spazio grigio, anche fuori dal frame
- le coordinate fuori frame vengono salvate e gestite correttamente

Controlli completi:
- **Click sinistro**: aggiungi angolo / trascina un angolo esistente
- **Click destro**: rimuovi l'angolo sotto il cursore
- **Rotella**: zoom centrato sul cursore
- **Tasto centrale trascinato** oppure **W/A/S/D**: pan
- **F**: fit, **R**: reset punti, **H**: mostra/nascondi aiuto
- **INVIO**: salva (servono 4 punti), **ESC**: annulla

Salva automaticamente `<nome_video>_roi.json` accanto al video
(es. `partita.mp4` → `partita_roi.json`).

### 2. Avvia il TUI
```
python smartcourt_tui_v3.py
```
- Opzione 1: analizza video (usa `<video>_roi.json` se presente, oppure `roi.json`)
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

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
- ✅ **Omografia del campo** (pixel -> metri reali)
- ✅ **In/out** (singolo, doppio, service box)
- ✅ **Velocità media tra rimbalzi** (km/h)
- ✅ **Metriche rally** (passaggi rete, durata, highlight score)
- ✅ **Calibrazione camera** da una vista (per velocità 3D sperimentale)

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
   - `smartcourt_metrics.py`
   - `court_geometry.py`
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

Per il campo singolo:
```
python smartcourt_roi_calibrate.py video.mp4 --court singles
```
Il file contiene anche omografia, misure del campo e calibrazione della
telecamera: serve per velocità, in/out e metriche.

### 2. Avvia il TUI
```
python smartcourt_tui_v3.py
```
- Opzione 1: analizza video (usa `<video>_roi.json` se presente, oppure `roi.json`)
- Opzione 2: ricalibra ROI
- Opzione 3: info modello

## Output
- `output/report.json` — statistiche
- `output/metrics.json` — rimbalzi, in/out, velocità, rally, highlight
- `output/tracks.jsonl` — traccia per-frame (palla + giocatori)
- `output/tracks_meta.json` — fps, rally, frame dei rimbalzi
- `output/heatmap_players.jpg` — heatmap giocatori
- `output/highlights.mp4` — highlights rally
- `output/frames/` — frame annotati

## Misurazione velocità: cosa è corretto e cosa no

L'omografia vale per punti **sul piano del campo**. La palla in volo è
sopra il campo, quindi il pixel proiettato sul piano non è la sua
posizione reale.

- ✅ **Velocità media tra due rimbalzi**: corretta, perché ai rimbalzi la
  palla è a quota zero.
- ⚠️ **Velocità istantanea in volo**: richiede il fit balistico 3D
  (funzione `CourtModel.fit_ballistic`) oppure due telecamere. È
  sperimentale e dipende dalla precisione della calibrazione.
- ❌ **Velocità = distanza pixel / tempo**: non usarla, sovrastima o
  sottostima a seconda dell'altezza della palla.

La telecamera deve essere **fissa** (treppiede). Con zoom/pan la
calibrazione del campo cambia e le metriche non sono valide.

## Cosa NON è ancora implementato
- Riconoscimento dritto/rovescio: serve pose/racchetta o un piccolo
  classificatore temporale addestrato su colpi etichettati.
- Rilevamento del punto (palla fuori, doppio rimbalzo, fallo): regole
  parziali presenti, ma da validare.
- Highlights con audio/tifo: per ora sono basati sulla durata del rally.
- Court Keypoints automatico (modello ResNet-50) — richiede integrazione specifica
- TrackNet ball tracking (modello U-Net 3-frame) — richiede preprocessing
- TensorRT — solo su Jetson

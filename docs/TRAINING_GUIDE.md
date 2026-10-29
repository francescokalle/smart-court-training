# Guida Training Step-by-Step

## Fase 1 — YOLO (Player/Racket/Ball Detection)

1. Apri `notebooks/01_train_yolo.ipynb` in Colab Pro.
2. Runtime → Change runtime type → GPU (T4 o A100).
3. Inserisci la tua API key Roboflow (crea account gratuito su roboflow.com).
4. Esegui tutte le celle.
5. Scarica `best.pt` e `best.onnx` sul tuo PC.
6. Copia i file sul Jetson in `/opt/smartcourt/models/`.

**Tempo stimato**: 1-2 ore su T4, 30 min su A100.

## Fase 2 — Court Keypoints

1. Usa lo stesso notebook YOLO ma con dataset court keypoints.
2. Modello: `yolov8n-pose.pt`.
3. Export ONNX con 14 keypoints.

**Tempo stimato**: 2-3 ore su T4.

## Fase 3 — TrackNet (Ball Tracking)

1. Apri `notebooks/02_train_tracknet.ipynb`.
2. Clona il repo TrackNetV3 per l'architettura completa:
   ```
   git clone https://github.com/qaz812345/TrackNetV3.git
   ```
3. Usa il loro `preprocess.py` per generare input/target.
4. Training: 30 epochs, ~2-3 ore su T4.
5. Export ONNX.

**Tempo stimato**: 3-4 ore su T4.

## Fase 4 — Deploy su Jetson

1. Copia i `.onnx` sul Jetson.
2. Converti con `trtexec`:
   ```bash
   trtexec --onnx=yolov8n_tennis.onnx --saveEngine=yolov8n_tennis.engine --fp16 --workspace=4096
   trtexec --onnx=tracknet.onnx --saveEngine=tracknet.engine --fp16 --workspace=4096
   ```
3. Configura massime prestazioni:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
4. Testa con video YouTube (validazione "a secco"):
   ```bash
   python scripts/validate.py --video test.mp4 --engine yolov8n_tennis.engine
   ```

## Checklist finale

- [ ] YOLO fine-tuned scaricato
- [ ] Court keypoints model scaricato
- [ ] TrackNet model scaricato
- [ ] Tutti i `.onnx` convertiti in `.engine`
- [ ] Benchmark latenza < 33ms (30 fps) su 2 cam
- [ ] Validazione su 5 video YouTube diversi

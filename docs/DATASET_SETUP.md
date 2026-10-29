# Setup Dataset su Google Drive (per Colab)

## 1. Struttura consigliata su Google Drive

```
MyDrive/smart-court-data/
├── tracknet/
│   ├── Dataset.zip          (da /home/server/datasets/tracknet/Dataset.zip)
│   ├── model_best.pt
│   └── ctb_regr_bounce.cbm
├── court_keypoints/
│   └── tennis_court_det_dataset.zip   (da /home/server/datasets/court_keypoints/)
├── courtside/
│   └── model.pt
└── videos_test/            (video YouTube scaricati per validazione)
```

## 2. Come caricare i dataset su Drive

### Opzione A — Upload diretto (consigliato)
1. Apri https://drive.google.com
2. Crea la cartella `smart-court-data`
3. Trascina i file `.zip` e `.pt` dalla tua macchina locale.

### Opzione B — Da questa macchina via rclone (se configurato)
```bash
rclone copy /home/server/datasets/tracknet/Dataset.zip gdrive:smart-court-data/tracknet/
rclone copy /home/server/datasets/court_keypoints/tennis_court_det_dataset.zip gdrive:smart-court-data/court_keypoints/
```

## 3. Come montare Drive in Colab

Nel notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = '/content/drive/MyDrive/smart-court-data'
```

## 4. Verifica spazio
- TrackNet zip: ~2.5 GB
- Court Keypoints zip: ~6.8 GB
- Totale: ~9.3 GB su Drive (piano gratuito ha 15 GB, sufficiente)

## 5. Video di test (validazione "a secco")
Scarica 3-5 video tennis amatoriali da YouTube:
```bash
pip install yt-dlp
yt-dlp -f "best[height<=720]" "URL_VIDEO" -o videos_test/%(title)s.%(ext)s
```
Metti i video in `smart-court-data/videos_test/`.

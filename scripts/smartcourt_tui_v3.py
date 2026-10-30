#!/usr/bin/env python3
"""Smart Court Tennis Analysis — TUI v3.

Integra:
- YOLO detection (player + ball)
- ByteTrack per tracking giocatori (ID persistenti)
- CourtROI: filtro geometrico campo (elimina spettatori)
- BallKalman: filtro di Kalman per la palla (elimina falsi positivi)
- detect_bounces: rilevamento rimbalzi
- Highlights FFmpeg + heatmap

Uso:
    python smartcourt_tui_v3.py

Prima calibra il campo (una volta per video):
    python smartcourt_roi_calibrate.py video.mp4
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from postprocessing import CourtROI, BallKalman, detect_bounces

try:
    from court_geometry import CourtModel
except ImportError:
    CourtModel = None

try:
    from smartcourt_metrics import analyze_tracks
except ImportError:
    analyze_tracks = None


# ── Utility ────────────────────────────────────────────────────────────────
def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print("=" * 62)
    print("  Smart Court Tennis Analysis — TUI v3")
    print("  YOLO + ByteTrack + Kalman + ROI + Bounce Detection")
    print("=" * 62)


def ask(prompt, default=None):
    val = input(f"{prompt} [{default}]: " if default else f"{prompt}: ").strip()
    return val if val else (default or "")


def run_cmd(cmd, desc=""):
    if desc:
        print(f"\n▶ {desc}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"  ❌ {e}")
        return False


def load_yolo(model_dir):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics non installato. Esegui: pip install ultralytics")
        return None
    for name in ("best.pt", "model.pt"):
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            print(f"  ✅ YOLO: {p}")
            return YOLO(p)
    print(f"  ❌ Nessun modello YOLO trovato in {model_dir}")
    return None


def roi_path_for(video_path):
    """ROI salvata accanto al video: partita.mp4 -> partita_roi.json."""
    p = Path(video_path)
    return p.with_name(p.stem + "_roi.json")


# ── Analisi video ───────────────────────────────────────────────────────────
def analyze_video(model, video_path, out_dir, roi_path=None):
    clear()
    banner()
    print(f"\n  Video:  {video_path}")
    print(f"  Output: {out_dir}")

    # Cerca prima la ROI specifica del video, poi il vecchio roi.json
    candidates = [roi_path] if roi_path else [roi_path_for(video_path), Path("roi.json")]
    roi = None
    used_roi_path = None
    for candidate in candidates:
        roi = CourtROI.from_json(candidate)
        if roi is not None:
            used_roi_path = candidate
            break
    if roi is not None:
        print(f"  ROI:    {used_roi_path} ({len(roi.polygon)} punti)")
    else:
        print("  ROI:    nessuna (tutte le persone saranno considerate player)")
        print("          Per filtrare gli spettatori: python smartcourt_roi_calibrate.py video.mp4")

    court_model = None
    if used_roi_path is not None and CourtModel is not None:
        try:
            court_model = CourtModel.from_roi_json(used_roi_path)
            cam = court_model.camera
            if cam is not None:
                print(f"  Campo:  {court_model.court_type} | camera h={cam.C[2]:.1f} m, "
                      f"f={cam.focal_px:.0f} px, conf={cam.confidence:.2f}")
            else:
                print(f"  Campo:  {court_model.court_type} | camera non calibrata")
        except Exception as exc:
            print(f"  Campo:  geometria non disponibile ({exc})")
            court_model = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Impossibile aprire: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  FPS: {fps:.1f} | Frame: {total} | {width}x{height}\n")

    tracks_path = out_dir / "tracks.jsonl"
    tracks_file = open(tracks_path, "w", encoding="utf-8")

    # Filtri
    MIN_PLAYER_H = 40
    MAX_PLAYER_H = 500
    MIN_BALL_AREA = 3
    MAX_BALL_AREA = 400

    kalman = BallKalman(max_jump=80.0)
    last_ball = None
    rally_start = None
    gap = 0
    rallies = []
    ball_positions = []      # (frame, x, y)
    ball_accepted = []       # (x, y) per bounce
    player_positions = []    # (x, y)
    player_tracks = {}       # id -> list of positions
    bounce_frames = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ByteTrack: tracking con ID persistenti
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            verbose=False, conf=0.25, iou=0.45,
        )
        dets = results[0].boxes
        has_ball = False
        ball_meas = None
        frame_players = []
        frame_ball_raw = None
        frame_ball = None

        if dets is not None:
            for b in dets:
                cls = int(b.cls.item())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                h = float(y2 - y1)
                w = float(x2 - x1)

                if cls == 0:  # person
                    if not (MIN_PLAYER_H <= h <= MAX_PLAYER_H):
                        continue
                    # ROI filter: scarta spettatori fuori campo
                    if roi is not None and not roi.contains((cx, cy), margin=0):
                        continue
                    player_positions.append((cx, cy))
                    tid = int(b.id.item()) if getattr(b, "id", None) is not None else -1
                    player_tracks.setdefault(tid, []).append((cx, cy))
                    frame_players.append({
                        "id": tid,
                        "bbox": [round(float(x1), 1), round(float(y1), 1),
                                 round(float(x2), 1), round(float(y2), 1)],
                        "center": [round(float(cx), 1), round(float(cy), 1)],
                    })

                elif cls == 32:  # sports ball
                    area = w * h
                    if not (MIN_BALL_AREA <= area <= MAX_BALL_AREA):
                        continue
                    frame_ball_raw = (cx, cy)
                    # filtro salto temporale
                    if last_ball is not None:
                        dist = float(np.hypot(cx - last_ball[0], cy - last_ball[1]))
                        if dist > 80.0:
                            continue
                    has_ball = True
                    ball_meas = (cx, cy)
                    last_ball = ball_meas

        # Kalman palla
        if ball_meas is not None:
            kx, ky, accepted = kalman.update(ball_meas)
            if accepted:
                ball_positions.append((frame_idx, kx, ky))
                ball_accepted.append((kx, ky))
                frame_ball = (kx, ky)
        else:
            kx, ky, accepted = kalman.update(None)
            if accepted and kalman.initialized:
                ball_positions.append((frame_idx, kx, ky))
                ball_accepted.append((kx, ky))
                frame_ball = (kx, ky)

        # Rally segmentation
        if has_ball:
            if rally_start is None:
                rally_start = frame_idx
            gap = 0
        else:
            if rally_start is not None:
                gap += 1
                if gap > 8:
                    if frame_idx - rally_start >= 15:
                        rallies.append((rally_start, frame_idx - gap))
                    rally_start = None
                    gap = 0
                    last_ball = None
                    kalman.reset()

        # Salva traccia per-frame (per metriche geometriche)
        tracks_file.write(json.dumps({
            "frame": frame_idx,
            "ball": [round(float(frame_ball[0]), 1), round(float(frame_ball[1]), 1)]
            if frame_ball is not None else None,
            "ball_raw": [round(float(frame_ball_raw[0]), 1), round(float(frame_ball_raw[1]), 1)]
            if frame_ball_raw is not None else None,
            "players": frame_players,
        }, separators=(",", ":")) + "\n")

        # Salva frame annotato
        if frame_idx % 30 == 0:
            annotated = results[0].plot()
            if roi is not None:
                roi.draw(annotated)
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"\r  Progresso: {frame_idx/total*100:.1f}% ({frame_idx}/{total})", end="", flush=True)

    cap.release()
    tracks_file.close()
    if rally_start is not None and frame_idx - rally_start >= 15:
        rallies.append((rally_start, frame_idx - gap))

    # Bounce detection
    if len(ball_accepted) > 3:
        bounce_idx = detect_bounces(
            ball_accepted,
            vy_threshold=4.0,
            min_interval=4,
            frames=[p[0] for p in ball_positions],
        )
        bounce_frames = [ball_positions[i][0] for i in bounce_idx if i < len(ball_positions)]
        print(f"\n\n  ✅ {frame_idx} frame | Rally: {len(rallies)} | Ball: {len(ball_positions)} | Bounce: {len(bounce_frames)} | Player: {len(player_positions)}")
    else:
        print(f"\n\n  ✅ {frame_idx} frame | Rally: {len(rallies)} | Ball: {len(ball_positions)} | Player: {len(player_positions)}")

    # Metadati tracce (per il modulo metriche)
    tracks_meta = {
        "video": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_idx,
        "bounce_frames": bounce_frames,
        "rallies": rallies,
    }
    (out_dir / "tracks_meta.json").write_text(
        json.dumps(tracks_meta, indent=2), encoding="utf-8"
    )

    # Metriche geometriche (solo se il campo è calibrato)
    metrics = None
    if court_model is not None and used_roi_path is not None and analyze_tracks is not None:
        print("\n📐 Calcolo metriche campo...")
        try:
            metrics = analyze_tracks(
                tracks_path,
                tracks_meta,
                used_roi_path,
                out_dir=out_dir,
                bounce_frames=bounce_frames,
                rallies=rallies,
            )
            ms = metrics["summary"]
            print(f"  Rimbalzi: {ms['bounces']} (in={ms['in_bounces']}, out={ms['out_bounces']})")
            print(f"  Velocità media: {ms['avg_speed_kmh']} km/h | max: {ms['max_speed_kmh']} km/h")
            print(f"  Rally: {ms['rallies']} | passaggi rete medi: {ms['avg_net_crossings']}")
            print(f"  Dettaglio: {out_dir / 'metrics.json'}")
        except Exception as exc:
            print(f"  ⚠️  Metriche non calcolate: {exc}")
    elif court_model is None:
        print("\n  ℹ️  Per velocità/in-out: ricalibra il campo con lo script ROI aggiornato.")

    # Report
    report = {
        "video": video_path,
        "fps": fps,
        "total_frames": frame_idx,
        "resolution": [width, height],
        "roi": roi.polygon.tolist() if roi is not None else None,
        "court_calibrated": court_model is not None,
        "rallies": len(rallies),
        "rally_list": rallies,
        "ball_detections": len(ball_positions),
        "bounces": len(bounce_frames),
        "bounce_frames": bounce_frames,
        "player_detections": len(player_positions),
        "player_tracks": len(player_tracks),
        "tracks": str(tracks_path),
        "metrics": metrics["summary"] if metrics else None,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {out_dir / 'report.json'}")

    # Heatmap player
    if player_positions:
        heat = np.zeros((height, width), dtype=np.float32)
        for cx, cy in player_positions:
            if 0 <= int(cx) < width and 0 <= int(cy) < height:
                heat[int(cy), int(cx)] += 1
        heat = cv2.GaussianBlur(heat, (0, 0), 30)
        if heat.max() > 0:
            heat = heat / heat.max()
        heat_img = (heat * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
        if roi is not None:
            roi.draw(heat_color, color=(255, 255, 255), thickness=2)
        cv2.imwrite(str(out_dir / "heatmap_players.jpg"), heat_color)
        print(f"  Heatmap: {out_dir / 'heatmap_players.jpg'}")

    # Highlights
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        has_ffmpeg = True
    except FileNotFoundError:
        has_ffmpeg = False
        print("  ⚠️  ffmpeg non trovato. Highlights saltati.")

    if has_ffmpeg and rallies:
        print("\n🎬 Generazione highlights...")
        seg_dir = out_dir / "segments"
        seg_dir.mkdir(exist_ok=True)
        seg_paths = []
        for i, (s, e) in enumerate(rallies):
            sp = seg_dir / f"seg_{i:03d}.mp4"
            dur = (e - s + 1) / fps
            run_cmd([
                "ffmpeg", "-y", "-ss", f"{s/fps:.3f}", "-i", video_path,
                "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-c:a", "aac", str(sp),
            ], desc=f"Seg {i+1}/{len(rallies)}")
            seg_paths.append(sp)

        if len(seg_paths) == 1:
            seg_paths[0].replace(out_dir / "highlights.mp4")
        elif len(seg_paths) > 1:
            lf = out_dir / "concat.txt"
            with open(lf, "w") as f:
                for sp in seg_paths:
                    f.write(f"file '{sp.resolve()}'\n")
            run_cmd([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lf),
                "-c", "copy", str(out_dir / "highlights.mp4"),
            ], desc="Concatenazione")
            lf.unlink(missing_ok=True)
        for sp in seg_paths:
            sp.unlink(missing_ok=True)
        try:
            seg_dir.rmdir()
        except OSError:
            pass
        print(f"\n  ✅ Highlights: {out_dir / 'highlights.mp4'}")

    print("\n  Premi INVIO per tornare al menu...")
    input()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    clear()
    banner()
    print()
    model_dir = ask("Cartella con i modelli", ".")
    model = load_yolo(model_dir)
    if model is None:
        input("\n  Premi INVIO per uscire...")
        return

    video_path = None
    out_dir = "output"

    while True:
        clear()
        banner()
        print()
        print("  1) Analizza video (tracking + Kalman + bounce + highlights)")
        print("  2) Calibra ROI campo (richiede video)")
        print("  3) Info modello")
        print("  4) Esci")
        print()
        c = input("  Scelta: ").strip()

        if c == "1":
            if not video_path:
                video_path = ask("Path del video")
                out_dir = ask("Cartella output", "output")
            analyze_video(model, video_path, out_dir)
        elif c == "2":
            v = ask("Path del video per calibrazione", video_path or "")
            if v:
                calib_script = Path(__file__).resolve().parent / "smartcourt_roi_calibrate.py"
                subprocess.run([sys.executable, str(calib_script), v])
                input("\n  Premi INVIO...")
        elif c == "3":
            clear()
            banner()
            print(f"\n  Classi YOLO: {model.names}")
            print(f"  Task: {model.task}")
            input("\n  Premi INVIO...")
        elif c == "4":
            print("\n  Arrivederci! 🎾\n")
            break
        else:
            print("  Scelta non valida.")
            input("  Premi INVIO...")


if __name__ == "__main__":
    main()

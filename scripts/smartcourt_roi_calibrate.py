#!/usr/bin/env python3
"""Calibrazione ROI del campo con zoom, pan e punti anche FUORI dal frame.

Serve per i video in cui gli angoli del campo sono tagliati dal bordo
dell'immagine: puoi dezoomare, cliccare nello spazio attorno al video e
salvare coordinate negative o maggiori della risoluzione.

Uso:
    python smartcourt_roi_calibrate.py video.mp4
    python smartcourt_roi_calibrate.py video.mp4 --court singles
    python smartcourt_roi_calibrate.py video.mp4 --frame 300
    python smartcourt_roi_calibrate.py video.mp4 --time 12.5

Il file salvato contiene anche omografia, misure del campo e calibrazione
della telecamera: serve al TUI per calcolare velocità, in/out e metriche.

Controlli:
    Click sinistro      aggiungi angolo (max 4)
    Trascina sinistro   sposta un angolo esistente
    Click destro        rimuovi l'angolo sotto il cursore (o l'ultimo)
    Rotella             zoom centrato sul cursore
    Trascina centrale   pan
    W / A / S / D       pan
    + / -               zoom
    F                   fit (adatta il frame alla finestra)
    R                   reset di tutti i punti
    H                   mostra/nascondi il pannello di aiuto
    INVIO               salva (servono 4 punti)
    ESC                 annulla

Output di default:
    <nome_video>_roi.json accanto al video.
    Esempio: partita.mp4 -> partita_roi.json

I punti sono salvati in coordinate immagine e possono essere negativi
o superiori alla risoluzione: il filtro ROI li gestisce correttamente.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from court_geometry import CourtModel
except Exception:  # pragma: no cover - il calibratore resta usabile anche senza
    CourtModel = None


def default_canvas() -> tuple[int, int]:
    """Canvas che sta nello schermo (tkinter è opzionale)."""
    screen_w, screen_h = 1920, 1080
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        pass
    width = min(1200, max(640, screen_w - 80))
    height = min(760, max(480, screen_h - 140))
    return width, height


def roi_path_for(video_path: str | Path) -> Path:
    p = Path(video_path)
    return p.with_name(p.stem + "_roi.json")


def segments_intersect(p1, p2, p3, p4) -> bool:
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


class RoiCalibrator:
    def __init__(self, frame: np.ndarray, video_path: str, frame_index: int,
                 canvas_w: int, canvas_h: int):
        self.frame = frame
        self.video_path = video_path
        self.frame_index = frame_index
        self.fh, self.fw = frame.shape[:2]
        self.canvas_w = int(canvas_w)
        self.canvas_h = int(canvas_h)

        self.points: list[tuple[float, float]] = []
        self.fit_zoom = min(self.canvas_w / self.fw, self.canvas_h / self.fh) * 0.82
        self.min_zoom = self.fit_zoom * 0.06
        self.max_zoom = self.fit_zoom * 30.0
        self.zoom = self.fit_zoom
        self.ox = (self.canvas_w - self.fw * self.zoom) / 2.0
        self.oy = (self.canvas_h - self.fh * self.zoom) / 2.0

        self.drag_point: int | None = None
        self.panning = False
        self.pan_start: tuple[int, int] | None = None
        self.pan_origin: tuple[float, float] | None = None
        self.mouse = (self.canvas_w // 2, self.canvas_h // 2)

        self.message = ""
        self.message_ttl = 0
        self.show_help = True
        self.result: list[tuple[float, float]] | None = None

    # ── Trasformazioni ─────────────────────────────────────────────────────
    def screen_to_image(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.ox) / self.zoom, (sy - self.oy) / self.zoom)

    def image_to_screen(self, ix: float, iy: float) -> tuple[float, float]:
        return (ix * self.zoom + self.ox, iy * self.zoom + self.oy)

    def fit(self) -> None:
        self.zoom = self.fit_zoom
        self.ox = (self.canvas_w - self.fw * self.zoom) / 2.0
        self.oy = (self.canvas_h - self.fh * self.zoom) / 2.0

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        ix, iy = self.screen_to_image(sx, sy)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self.zoom = new_zoom
        self.ox = sx - ix * self.zoom
        self.oy = sy - iy * self.zoom

    def pan(self, dx: float, dy: float) -> None:
        self.ox += dx
        self.oy += dy

    def nearest_point(self, sx: float, sy: float, radius: float = 14.0) -> int | None:
        best_idx = None
        best_dist = radius
        for i, (ix, iy) in enumerate(self.points):
            px, py = self.image_to_screen(ix, iy)
            dist = math.hypot(px - sx, py - sy)
            if dist <= best_dist:
                best_idx = i
                best_dist = dist
        return best_idx

    def undo(self) -> None:
        if self.points:
            removed = self.points.pop()
            self._say(f"Rimosso punto {removed[0]:.0f},{removed[1]:.0f}")

    def _say(self, msg: str, ttl: int = 150) -> None:
        self.message = msg
        self.message_ttl = ttl
        print(f"  {msg}")

    # ── Eventi mouse ───────────────────────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param) -> None:
        prev_mouse = self.mouse
        self.mouse = (x, y)

        if event == cv2.EVENT_MOUSEWHEEL:
            # Alcune versioni di OpenCV riportano (0,0) per la rotella:
            # in quel caso usa l'ultima posizione nota del cursore.
            zx, zy = (x, y) if (x, y) != (0, 0) else prev_mouse
            self.mouse = (zx, zy)
            factor = 1.2 if flags > 0 else 1.0 / 1.2
            self.zoom_at(zx, zy, factor)
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start = (x, y)
            self.pan_origin = (self.ox, self.oy)
            return

        if event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            return

        if event == cv2.EVENT_MOUSEMOVE and self.panning:
            # Se il tasto centrale è stato rilasciato fuori dalla finestra,
            # OpenCV potrebbe non inviare MBUTTONUP: controlla i flag.
            if not (flags & cv2.EVENT_FLAG_MBUTTON):
                self.panning = False
                return
            if self.pan_start is not None and self.pan_origin is not None:
                self.ox = self.pan_origin[0] + (x - self.pan_start[0])
                self.oy = self.pan_origin[1] + (y - self.pan_start[1])
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.nearest_point(x, y)
            if idx is not None:
                self.drag_point = idx
                self._say(f"Sposto punto {idx + 1}")
            elif len(self.points) < 4:
                ix, iy = self.screen_to_image(x, y)
                self.points.append((ix, iy))
                self.drag_point = len(self.points) - 1
                where = "dentro" if (0 <= ix < self.fw and 0 <= iy < self.fh) else "FUORI frame"
                self._say(f"Punto {len(self.points)}: ({ix:.1f}, {iy:.1f}) [{where}]")
            else:
                self._say("Hai già 4 punti: trascinali per spostarli o usa il tasto destro.")
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.drag_point = None
            return

        if event == cv2.EVENT_MOUSEMOVE and self.drag_point is not None:
            if not (flags & cv2.EVENT_FLAG_LBUTTON):
                self.drag_point = None
                return
            self.points[self.drag_point] = self.screen_to_image(x, y)
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            idx = self.nearest_point(x, y)
            if idx is not None:
                removed = self.points.pop(idx)
                self._say(f"Rimosso punto {idx + 1} ({removed[0]:.0f},{removed[1]:.0f})")
            else:
                self.undo()

    # ── Rendering ──────────────────────────────────────────────────────────
    @staticmethod
    def _blit(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
        h, w = img.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            return
        canvas[y0:y1, x0:x1] = img[y0 - y:y1 - y, x0 - x:x1 - x]

    def render(self) -> np.ndarray:
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 38, np.uint8)

        # Disegna solo la porzione di frame visibile (efficiente anche a zoom alto)
        ix0 = (0 - self.ox) / self.zoom
        iy0 = (0 - self.oy) / self.zoom
        ix1 = (self.canvas_w - self.ox) / self.zoom
        iy1 = (self.canvas_h - self.oy) / self.zoom
        sx0 = max(0, int(math.floor(ix0)))
        sy0 = max(0, int(math.floor(iy0)))
        sx1 = min(self.fw, int(math.ceil(ix1)))
        sy1 = min(self.fh, int(math.ceil(iy1)))

        if sx1 > sx0 and sy1 > sy0:
            crop = self.frame[sy0:sy1, sx0:sx1]
            dst_w = max(1, int(round((sx1 - sx0) * self.zoom)))
            dst_h = max(1, int(round((sy1 - sy0) * self.zoom)))
            interp = cv2.INTER_AREA if self.zoom < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)
            dx = int(round(sx0 * self.zoom + self.ox))
            dy = int(round(sy0 * self.zoom + self.oy))
            self._blit(canvas, resized, dx, dy)

        # Bordo del frame originale
        fx0 = int(round(self.ox))
        fy0 = int(round(self.oy))
        fx1 = int(round(self.ox + self.fw * self.zoom))
        fy1 = int(round(self.oy + self.fh * self.zoom))
        cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), (85, 85, 85), 1, cv2.LINE_AA)

        screen_pts = [self.image_to_screen(ix, iy) for ix, iy in self.points]

        # Riempimento leggero del poligono
        if len(screen_pts) >= 3:
            pts_i = np.array([[int(round(px)), int(round(py))] for px, py in screen_pts], np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts_i], (0, 220, 0))
            cv2.addWeighted(overlay, 0.14, canvas, 0.86, 0, canvas)

        # Linee del poligono
        if len(screen_pts) >= 2:
            pts_i = np.array([[int(round(px)), int(round(py))] for px, py in screen_pts], np.int32)
            cv2.polylines(canvas, [pts_i], len(screen_pts) == 4, (0, 255, 0), 2, cv2.LINE_AA)

        # Punti + coordinate
        for i, ((ix, iy), (px, py)) in enumerate(zip(self.points, screen_pts)):
            inside = 0 <= ix < self.fw and 0 <= iy < self.fh
            color = (60, 220, 60) if inside else (0, 165, 255)
            center = (int(round(px)), int(round(py)))
            cv2.circle(canvas, center, 8, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 7, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 7, (255, 255, 255), 1, cv2.LINE_AA)

            label = f"{i + 1} ({ix:.0f},{iy:.0f})"
            tx, ty = center[0] + 12, center[1] - 10
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Mirino sul cursore
        mx, my = self.mouse
        if 0 <= mx < self.canvas_w and 0 <= my < self.canvas_h:
            cv2.line(canvas, (mx - 10, my), (mx + 10, my), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (mx, my - 10), (mx, my + 10), (255, 255, 0), 1, cv2.LINE_AA)

        self._draw_status(canvas)
        return canvas

    def _draw_status(self, canvas: np.ndarray) -> None:
        if not self.show_help:
            self._draw_message(canvas)
            return
        lines = [
            "SMART COURT — 4 angoli | se tagliati: dezoom e clicca nel grigio",
            "Click SX: aggiungi/sposta | Destro: rimuovi | Rotella: zoom | Centrale o WASD: pan",
            "F: fit | R: reset | H: nascondi aiuto | INVIO: salva | ESC: annulla",
            f"Punti: {len(self.points)}/4 | Zoom: {self.zoom / self.fit_zoom * 100:.0f}%"
            + ("  |  ✅ pronto, premi INVIO" if len(self.points) == 4 else ""),
        ]
        pad = 8
        line_h = 22
        box_h = len(lines) * line_h + 2 * pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (8, 8), (self.canvas_w - 8, 8 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)

        for i, line in enumerate(lines):
            y = 8 + pad + 17 + i * line_h
            cv2.putText(canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._draw_message(canvas)

    def _draw_message(self, canvas: np.ndarray) -> None:

        if self.message and self.message_ttl > 0:
            y = self.canvas_h - 18
            cv2.putText(canvas, self.message, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, self.message, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 1, cv2.LINE_AA)
            self.message_ttl -= 1

    # ── Loop ───────────────────────────────────────────────────────────────
    def _validate(self) -> bool:
        if len(self.points) != 4:
            self._say(f"Servono 4 punti, ne hai {len(self.points)}")
            return False

        p = self.points
        if segments_intersect(p[0], p[1], p[2], p[3]) or segments_intersect(p[1], p[2], p[3], p[0]):
            self._say("⚠️ Poligono incrociato: controlla l'ordine dei 4 angoli.")
            return False

        area = abs(cv2.contourArea(np.array(p, dtype=np.float32)))
        if area < 100:
            self._say("⚠️ Area troppo piccola: i punti sono quasi sovrapposti.")
            return False
        return True

    def run(self) -> list[tuple[float, float]] | None:
        win = "SmartCourt — Calibrazione ROI"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, self.on_mouse)

        while True:
            cv2.imshow(win, self.render())
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                return None

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None

            if key in (13, 10):  # INVIO
                if self._validate():
                    self.result = list(self.points)
                    break
                continue

            if key in (ord("r"), ord("R")):
                self.points.clear()
                self._say("Reset punti.")
            elif key in (ord("f"), ord("F")):
                self.fit()
                self._say("Fit finestra.")
            elif key in (ord("u"), ord("U"), 8, 127):
                self.undo()
            elif key in (ord("h"), ord("H")):
                self.show_help = not self.show_help
                self._say("Aiuto nascosto (H per mostrarlo)." if not self.show_help else "Aiuto visibile.")
            elif key in (ord("w"), ord("W")):
                self.pan(0, 40)
            elif key in (ord("s"), ord("S")):
                self.pan(0, -40)
            elif key in (ord("a"), ord("A")):
                self.pan(40, 0)
            elif key in (ord("d"), ord("D")):
                self.pan(-40, 0)
            elif key in (ord("+"), ord("=")):
                self.zoom_at(self.canvas_w / 2, self.canvas_h / 2, 1.2)
            elif key in (ord("-"), ord("_")):
                self.zoom_at(self.canvas_w / 2, self.canvas_h / 2, 1 / 1.2)

        cv2.destroyAllWindows()
        return self.result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrazione ROI con zoom/pan e punti fuori frame")
    parser.add_argument("video", help="Path del video")
    parser.add_argument("--frame", type=int, default=0, help="Frame da usare (default 0)")
    parser.add_argument("--time", type=float, default=None,
                        help="Secondo del video da usare (alternativa a --frame)")
    parser.add_argument("--out", default=None,
                        help="File output (default: <video>_roi.json accanto al video)")
    parser.add_argument("--court", choices=("doubles", "singles"), default="doubles",
                        help="Righe da cliccare: doppio (default) o singolo")
    parser.add_argument("--canvas-w", type=int, default=0, help="Larghezza finestra (default auto)")
    parser.add_argument("--canvas-h", type=int, default=0, help="Altezza finestra (default auto)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Impossibile aprire il video: {args.video}")
        sys.exit(1)

    if args.time is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000.0)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ok, frame = cap.read()
    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok or frame is None:
        print("❌ Impossibile leggere il frame richiesto.")
        sys.exit(1)

    canvas_w, canvas_h = default_canvas()
    if args.canvas_w > 0:
        canvas_w = args.canvas_w
    if args.canvas_h > 0:
        canvas_h = args.canvas_h

    court_label = "doppio (righe esterne)" if args.court == "doubles" else "singolo"
    print("=" * 64)
    print(f"  CALIBRAZIONE ROI — campo {court_label}")
    print("=" * 64)
    print("  Ordine: alto-sx → alto-dx → basso-dx → basso-sx")
    print("  (alto = lato lontano, basso = lato vicino alla telecamera)")
    print()
    print("  Se un angolo è tagliato dal bordo del video:")
    print("   1. usa la rotella per DEZOOMARE")
    print("   2. clicca nello spazio grigio attorno al video")
    print("   3. le coordinate fuori frame vengono salvate e gestite")
    print()
    print("  Rotella: zoom | Centrale o WASD: pan | F: fit")
    print("  Click destro: rimuovi punto | R: reset | INVIO: salva | ESC: annulla")
    print("=" * 64)

    calib = RoiCalibrator(frame, args.video, actual_frame, canvas_w, canvas_h)
    points = calib.run()
    if points is None:
        print("\nCalibrazione annullata.")
        return

    out_path = Path(args.out) if args.out else roi_path_for(args.video)
    data = {
        "video": str(args.video),
        "frame": actual_frame,
        "note": "Coordinate in pixel immagine; possono essere fuori frame.",
    }
    if CourtModel is not None:
        try:
            cm = CourtModel(points, court_type=args.court,
                            image_size=(calib.fw, calib.fh))
            data.update(cm.to_dict())
        except Exception as exc:
            print(f"\n⚠️  Geometria campo non calcolata: {exc}")
            data.update({
                "court": args.court,
                "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in points],
                "frame_size": [int(calib.fw), int(calib.fh)],
            })
    else:
        data.update({
            "court": args.court,
            "polygon": [[round(float(x), 1), round(float(y), 1)] for x, y in points],
            "frame_size": [int(calib.fw), int(calib.fh)],
        })

    # Mantieni i campi base sempre presenti
    data["video"] = str(args.video)
    data["frame"] = actual_frame
    data["polygon"] = [[round(float(x), 1), round(float(y), 1)] for x, y in points]
    try:
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        if args.out:
            print(f"\n❌ Impossibile scrivere {out_path}: {exc}")
            return
        # Es. video su disco protetto/network: salva nella cartella corrente
        out_path = Path("roi.json")
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\n⚠️  Non ho potuto scrivere accanto al video ({exc}).")
        print(f"   ROI salvata in: {out_path.resolve()}")

    print(f"\n✅ ROI salvata in: {out_path}")
    for i, (x, y) in enumerate(points, 1):
        inside = 0 <= x < calib.fw and 0 <= y < calib.fh
        print(f"   {i}: ({x:.1f}, {y:.1f})  {'dentro' if inside else 'FUORI frame'}")

    cam = data.get("camera")
    if cam:
        print(f"  📷 Camera: h={cam['center_m'][2]:.1f} m, "
              f"f={cam['focal_px']:.0f} px, conf={cam['confidence']:.2f}")
        print("     → velocità 3D, in/out e metriche disponibili.")
    else:
        print("  ⚠️  Camera non calibrata: in/out sì, velocità 3D no.")

    print(f"\n  Ora avvia il TUI: python smartcourt_tui_v3.py")


if __name__ == "__main__":
    main()

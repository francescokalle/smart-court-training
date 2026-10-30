#!/usr/bin/env python3
"""Calibrazione ROI: clicca i 4 angoli del campo su un frame del video.

Uso:
    python smartcourt_roi_calibrate.py video.mp4

Output:
    roi.json con i 4 punti (in ordine: alto-sx, alto-dx, basso-dx, basso-sx)

Suggerimento: clicca gli angoli esterni del campo (singolo o doppio).
"""
import argparse
import json
import sys

import cv2

points = []


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"  Punto {len(points)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path del video")
    parser.add_argument("--frame", type=int, default=0, help="Frame da usare (default 0)")
    parser.add_argument("--out", default="roi.json", help="File output")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire {args.video}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Errore: impossibile leggere il frame")
        sys.exit(1)

    print("=" * 60)
    print("  CALIBRAZIONE ROI")
    print("=" * 60)
    print("  Clicca i 4 angoli del campo in questo ordine:")
    print("    1. Alto-sinistra")
    print("    2. Alto-destra")
    print("    3. Basso-destra")
    print("    4. Basso-sinistra")
    print("  Premi ESC per annullare, INVIO per salvare.")
    print("=" * 60)

    win = "Calibrazione ROI - clicca 4 angoli"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, click)

    while True:
        display = frame.copy()
        for i, p in enumerate(points):
            cv2.circle(display, p, 6, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(display, points[i], points[i + 1], (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(display, points[3], points[0], (0, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            print("Annullato.")
            cv2.destroyAllWindows()
            return
        if key == 13:  # INVIO
            if len(points) == 4:
                break
            else:
                print(f"  Servono 4 punti, ne hai {len(points)}")

    cv2.destroyAllWindows()
    data = {"polygon": points, "video": args.video, "frame": args.frame}
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ ROI salvata in {args.out}")
    print(f"   Punti: {points}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smart Court — metriche geometriche da tracks.jsonl.

Legge il file prodotto dal TUI (una riga JSON per frame) e calcola:
- rimbalzi in coordinate campo (metri)
- in/out (singolo, doppio, service box)
- velocità media tra rimbalzi (km/h) — fisicamente corretta
- attraversamenti della rete e statistiche rally
- highlight score e classifica dei rally
- (opzionale) velocità stimata con fit balistico 3D

Uso:
    python smartcourt_metrics.py output/tracks.jsonl output/tracks_meta.json video_roi.json

Output:
    output/metrics.json + riepilogo a console.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from court_geometry import (
    COURT_LENGTH,
    DOUBLES_WIDTH,
    NET_Y,
    SINGLES_LEFT,
    SINGLES_RIGHT,
    CourtModel,
)


def load_tracks(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_frame(records: Iterable[dict]) -> dict[int, dict]:
    return {int(r["frame"]): r for r in records if "frame" in r}


def nearest_ball(
    by_frame: dict[int, dict],
    frame: int,
    max_distance_frames: int = 3,
) -> tuple[list[float] | None, int | None]:
    for d in range(max_distance_frames + 1):
        for f in (frame - d, frame + d):
            rec = by_frame.get(f)
            if rec and rec.get("ball"):
                return rec["ball"], f
    return None, None


def detect_bounces(
    records: list[dict],
    fps: float,
    vy_threshold: float = 4.0,
    min_interval_frames: int = 8,
) -> list[int]:
    """Fallback: rileva rimbalzi dall'inversione della velocità y."""
    pts = [
        (int(r["frame"]), float(r["ball"][0]), float(r["ball"][1]))
        for r in records
        if r.get("ball")
    ]
    bounces = []
    last = -10**9
    for i in range(1, len(pts) - 1):
        v_prev = pts[i][2] - pts[i - 1][2]
        v_next = pts[i + 1][2] - pts[i][2]
        if v_prev > vy_threshold and v_next < -vy_threshold and (pts[i][0] - last) >= min_interval_frames:
            bounces.append(pts[i][0])
            last = pts[i][0]
    return bounces


def count_net_crossings(
    court_points: list[list[float] | None],
    hysteresis_m: float = 0.25,
    min_frames_between: int = 4,
) -> int:
    """Conta quante volte la palla passa da un lato all'altro della rete."""
    side = None
    crossings = 0
    last_cross_frame = -10**9
    for i, p in enumerate(court_points):
        if p is None:
            continue
        y = float(p[1])
        if y < NET_Y - hysteresis_m:
            new_side = -1
        elif y > NET_Y + hysteresis_m:
            new_side = 1
        else:
            continue
        if side is not None and new_side != side and (i - last_cross_frame) >= min_frames_between:
            crossings += 1
            last_cross_frame = i
        side = new_side
    return crossings


def _merge_bounces(
    bounces: list[dict],
    fps: float,
    min_gap_s: float = 0.18,
    min_distance_m: float = 0.8,
) -> list[dict]:
    """Unisce rimbalzi duplicati ravvicinati (stesso impatto)."""
    merged: list[dict] = []
    for b in sorted(bounces, key=lambda x: x["frame"]):
        if not merged:
            merged.append(b)
            continue
        prev = merged[-1]
        dt = (b["frame"] - prev["frame"]) / fps
        dist = math.hypot(
            b["court"][0] - prev["court"][0],
            b["court"][1] - prev["court"][1],
        )
        if dt < min_gap_s and dist < min_distance_m:
            # tieni il rimbalzo con inversione verticale più netta, se disponibile
            if abs(b.get("vy_reversal", 0)) > abs(prev.get("vy_reversal", 0)):
                merged[-1] = b
            continue
        merged.append(b)
    return merged


def analyze_tracks(
    tracks_path: str | Path,
    meta: dict,
    roi_json: str | Path,
    out_dir: str | Path | None = None,
    bounce_frames: list[int] | None = None,
    rallies: list[list[int]] | None = None,
) -> dict:
    """Calcola tutte le metriche geometriche.

    meta deve contenere almeno: fps, width, height.
    roi_json è il file salvato dal calibratore con i 4 angoli del campo.
    """
    fps = float(meta.get("fps") or 30.0)
    model = CourtModel.from_roi_json(roi_json)
    records = load_tracks(tracks_path)
    by_frame = index_by_frame(records)
    if not records:
        raise ValueError("tracks.jsonl è vuoto")

    if bounce_frames is None:
        bounce_frames = meta.get("bounce_frames") or detect_bounces(records, fps)
    if rallies is None:
        rallies = meta.get("rallies") or []

    # ── Rimbalzi ───────────────────────────────────────────────────────────
    bounces: list[dict] = []
    for bf in bounce_frames:
        ball, actual_frame = nearest_ball(by_frame, int(bf))
        if ball is None or actual_frame is None:
            continue
        court = model.image_to_court([ball])[0]
        ball_y = float(ball[1])
        prev_rec = by_frame.get(actual_frame - 1)
        next_rec = by_frame.get(actual_frame + 1)
        vy_reversal = 0.0
        if prev_rec and prev_rec.get("ball") and next_rec and next_rec.get("ball"):
            vy_reversal = float(next_rec["ball"][1] - prev_rec["ball"][1])
        bounces.append(
            {
                "frame": int(actual_frame),
                "image": [round(float(ball[0]), 1), round(float(ball[1]), 1)],
                "court": [round(float(court[0]), 2), round(float(court[1]), 2)],
                "in_singles": bool(model.is_in_court(court, "singles")),
                "in_doubles": bool(model.is_in_court(court, "doubles")),
                "in_service_near": bool(model.is_in_court(court, "service_near")),
                "in_service_far": bool(model.is_in_court(court, "service_far")),
                "line_distance_m": round(model.nearest_line_distance_m(court), 2),
                "vy_reversal": round(float(vy_reversal), 2),
            }
        )
    bounces = _merge_bounces(bounces, fps)

    # ── Velocità tra rimbalzi ──────────────────────────────────────────────
    speeds: list[dict] = []
    for b1, b2 in zip(bounces, bounces[1:]):
        dt = (b2["frame"] - b1["frame"]) / fps
        dist = model.distance_m(b1["court"], b2["court"])
        if not (0.15 <= dt <= 4.0):
            continue
        if dist > 35.0:
            continue
        same_side = (b1["court"][1] < NET_Y) == (b2["court"][1] < NET_Y)
        kmh = model.speed_kmh(dist, dt)
        speeds.append(
            {
                "from_frame": b1["frame"],
                "to_frame": b2["frame"],
                "distance_m": round(float(dist), 2),
                "dt_s": round(float(dt), 3),
                "speed_kmh": round(float(kmh), 1),
                "same_side": bool(same_side),
                "valid": bool(not same_side),
            }
        )

    valid_speeds = [s["speed_kmh"] for s in speeds if s["valid"]]
    avg_speed = float(np.mean(valid_speeds)) if valid_speeds else None
    max_speed = float(np.max(valid_speeds)) if valid_speeds else None

    # ── Rally / highlight ──────────────────────────────────────────────────
    rally_stats: list[dict] = []
    for idx, (start, end) in enumerate(rallies):
        start, end = int(start), int(end)
        court_points: list[list[float] | None] = []
        for f in range(start, end + 1):
            rec = by_frame.get(f)
            if rec and rec.get("ball"):
                court_points.append(model.image_to_court([rec["ball"]])[0].tolist())
            else:
                court_points.append(None)
        crossings = count_net_crossings(court_points)
        n_bounces = sum(1 for b in bounces if start <= b["frame"] <= end)
        rally_speeds = [
            s["speed_kmh"]
            for s in speeds
            if start <= s["from_frame"] <= end and s["valid"]
        ]
        duration = max(0.0, (end - start) / fps)
        near_line = min(
            (b["line_distance_m"] for b in bounces if start <= b["frame"] <= end),
            default=99.0,
        )
        score = crossings * 10.0 + n_bounces * 2.0 + duration
        if near_line < 1.0:
            score += (1.0 - near_line) * 5.0
        rally_stats.append(
            {
                "index": idx,
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start / fps, 2),
                "end_s": round(end / fps, 2),
                "duration_s": round(duration, 2),
                "net_crossings": crossings,
                "bounces": n_bounces,
                "max_speed_kmh": round(max(rally_speeds), 1) if rally_speeds else None,
                "nearest_line_m": round(float(near_line), 2) if near_line < 99 else None,
                "highlight_score": round(score, 1),
            }
        )

    rally_stats.sort(key=lambda r: r["highlight_score"], reverse=True)

    in_key = "in_doubles" if model.court_type == "doubles" else "in_singles"
    in_count = sum(1 for b in bounces if b[in_key])
    out_count = len(bounces) - in_count

    summary = {
        "court_type": model.court_type,
        "camera_confidence": round(model.camera.confidence, 3) if model.camera else None,
        "camera_height_m": round(float(model.camera.C[2]), 2) if model.camera else None,
        "focal_px": round(float(model.camera.focal_px), 1) if model.camera else None,
        "bounces": len(bounces),
        "in_bounces": int(in_count),
        "out_bounces": int(out_count),
        "in_ratio": round(in_count / len(bounces), 3) if bounces else None,
        "avg_speed_kmh": round(avg_speed, 1) if avg_speed is not None else None,
        "max_speed_kmh": round(max_speed, 1) if max_speed is not None else None,
        "speed_samples": len(valid_speeds),
        "rallies": len(rally_stats),
        "avg_net_crossings": round(
            float(np.mean([r["net_crossings"] for r in rally_stats])), 2
        )
        if rally_stats
        else None,
        "top_highlights": rally_stats[:5],
    }

    result = {
        "summary": summary,
        "bounces": bounces,
        "speeds": speeds,
        "rallies": rally_stats,
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Metriche geometriche Smart Court")
    parser.add_argument("tracks", help="output/tracks.jsonl")
    parser.add_argument("meta", help="output/tracks_meta.json")
    parser.add_argument("roi", help="file ROI con i 4 angoli del campo")
    parser.add_argument("--out", default=None, help="cartella output (default: cartella tracks)")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else Path(args.tracks).parent
    result = analyze_tracks(args.tracks, meta, args.roi, out_dir=out_dir)
    s = result["summary"]

    print("=" * 62)
    print("  METRICHE CAMPO")
    print("=" * 62)
    print(f"  Campo:            {s['court_type']}")
    print(f"  Camera:           h={s['camera_height_m']} m, f={s['focal_px']} px, "
          f"conf={s['camera_confidence']}")
    print(f"  Rimbalzi:         {s['bounces']}  (in={s['in_bounces']}, out={s['out_bounces']})")
    print(f"  Velocità media:   {s['avg_speed_kmh']} km/h  "
          f"(max {s['max_speed_kmh']} km/h, {s['speed_samples']} tratte)")
    print(f"  Rally:            {s['rallies']}  "
          f"(passaggi rete medi: {s['avg_net_crossings']})")
    print("  Top highlight:")
    for r in s["top_highlights"][:3]:
        print(f"    #{r['index']}  {r['start_s']}s-{r['end_s']}s  "
              f"rete={r['net_crossings']}  velocità max={r['max_speed_kmh']} km/h")
    print("=" * 62)
    print(f"  Dettaglio: {Path(out_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()

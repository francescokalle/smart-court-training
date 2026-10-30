"""Smart Court — Post-processing per detection tennis.

Include:
- CourtROI: filtro geometrico (maschera campo)
- BallKalman: filtro di Kalman per traiettoria palla
- detect_bounces: rilevamento rimbalzi
- filter_ball_detections: filtri dimensione + salto temporale
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class CourtROI:
    """Poligono del campo in coordinate pixel. Filtra detection fuori campo."""

    def __init__(self, polygon: Sequence[Tuple[float, float]]):
        if len(polygon) < 3:
            raise ValueError("Il poligono ROI richiede almeno 3 punti")
        self.polygon = np.array(polygon, dtype=np.float32)
        self._contour = self.polygon.reshape(-1, 1, 2).astype(np.int32)

    def contains(self, point: Tuple[float, float], margin: int = 0) -> bool:
        """True se il punto è dentro il poligono (margin in pixel)."""
        if margin:
            # approssimazione: espandi il poligono verso il centro
            center = self.polygon.mean(axis=0)
            expanded = center + (self.polygon - center) * (1 + margin / max(1.0, np.linalg.norm(self.polygon - center, axis=1).mean()))
            contour = expanded.reshape(-1, 1, 2).astype(np.int32)
            return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0
        return cv2.pointPolygonTest(self._contour, (float(point[0]), float(point[1])), False) >= 0

    def draw(self, frame: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        cv2.polylines(frame, [self._contour], True, color, thickness)
        return frame

    @classmethod
    def from_json(cls, path: str | Path) -> Optional["CourtROI"]:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        poly = data.get("polygon") or data.get("image_corners")
        if not poly:
            return None
        return cls(poly)

    def to_json(self, path: str | Path, extra: Optional[dict] = None) -> None:
        data = {"polygon": self.polygon.tolist()}
        if extra:
            data.update(extra)
        Path(path).write_text(json.dumps(data, indent=2))


class BallKalman:
    """Kalman filter 2D per la palla (posizione + velocità)."""

    def __init__(self, max_jump: float = 80.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.initialized = False
        self.max_jump = max_jump
        self.last_pos: Optional[Tuple[float, float]] = None

    def update(self, meas: Optional[Tuple[float, float]]) -> Tuple[float, float, bool]:
        """Ritorna (x, y, accepted). accepted=False se la misura è un outlier."""
        if not self.initialized:
            if meas is None:
                return (0.0, 0.0, False)
            self.kf.statePost = np.array([[meas[0]], [meas[1]], [0], [0]], dtype=np.float32)
            self.initialized = True
            self.last_pos = meas
            return (meas[0], meas[1], True)

        pred = self.kf.predict()
        accepted = False
        if meas is not None:
            # filtro salto fisico
            if self.last_pos is not None:
                dist = float(np.hypot(meas[0] - self.last_pos[0], meas[1] - self.last_pos[1]))
                if dist > self.max_jump:
                    meas = None
            if meas is not None:
                self.kf.correct(np.array([[meas[0]], [meas[1]]], dtype=np.float32))
                accepted = True
                self.last_pos = meas
        state = self.kf.statePost
        return (float(state[0]), float(state[1]), accepted)

    def reset(self):
        self.initialized = False
        self.last_pos = None


def filter_ball_detections(
    detections: List[dict],
    last_pos: Optional[Tuple[float, float]] = None,
    max_jump: float = 80.0,
    min_area: float = 3.0,
    max_area: float = 400.0,
) -> List[dict]:
    """Filtra detection palla per area e salto temporale."""
    out = []
    prev = last_pos
    for d in detections:
        area = d.get("area", 0)
        if not (min_area <= area <= max_area):
            continue
        cx, cy = d["cx"], d["cy"]
        if prev is not None:
            dist = float(np.hypot(cx - prev[0], cy - prev[1]))
            if dist > max_jump:
                continue
        out.append(d)
        prev = (cx, cy)
    return out


def detect_bounces(
    positions: List[Tuple[float, float]],
    vy_threshold: float = 4.0,
    min_interval: int = 4,
    frames: Optional[Sequence[int]] = None,
) -> List[int]:
    """Rileva rimbalzi come inversione brusca della velocità verticale (down -> up).

    Se `frames` è fornito, la velocità viene normalizzata per il salto
    temporale tra campioni (utile quando il tracker salta qualche frame).
    `min_interval` è in frame.
    """
    bounces = []
    last = -10**9
    for i in range(1, len(positions) - 1):
        if frames is not None:
            dt_prev = frames[i] - frames[i - 1]
            dt_next = frames[i + 1] - frames[i]
        else:
            dt_prev = dt_next = 1
        if dt_prev <= 0 or dt_next <= 0:
            continue
        v_prev = (positions[i][1] - positions[i - 1][1]) / dt_prev
        v_next = (positions[i + 1][1] - positions[i][1]) / dt_next
        f = frames[i] if frames is not None else i
        if v_prev > vy_threshold and v_next < -vy_threshold and (f - last) >= min_interval:
            bounces.append(i)
            last = f
    return bounces


def map_to_court(H: np.ndarray, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Mappa punti immagine -> coordinate campo (metri)."""
    if not points:
        return []
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]

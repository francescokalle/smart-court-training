"""Smart Court — geometria del campo, omografia e fisica della palla.

Questo modulo risponde a una domanda precisa: dove si trova la palla
nello spazio reale, e quanto va veloce?

Contiene:
- misure ufficiali del campo da tennis (singolo e doppio)
- omografia immagine -> coordinate campo (metri)
- calibrazione della telecamera da una sola vista del campo
- conversione di un punto immagine sul piano del campo
- test in/out (singolo, doppio, service box)
- velocità media tra due rimbalzi (fisicamente corretta)
- fit balistico 3D della traiettoria (velocità istantanea stimata)

Nota importante:
l'omografia vale per punti che stanno SUL PIANO DEL CAMPO. La palla in
volo è sopra il campo: il punto immagine proiettato sul piano non è la
sua posizione reale. Per questo:
- ai rimbalzi la posizione è corretta (la palla tocca terra)
- la velocità media tra due rimbalzi è corretta
- la velocità istantanea in volo richiede il fit 3D (o due telecamere)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# ── Misure ufficiali (metri) ────────────────────────────────────────────────
COURT_LENGTH = 23.77
DOUBLES_WIDTH = 10.97
SINGLES_WIDTH = 8.23
NET_Y = COURT_LENGTH / 2.0                 # 11.885
SERVICE_LINE_OFFSET = 6.40                 # dalla rete
SINGLES_LEFT = (DOUBLES_WIDTH - SINGLES_WIDTH) / 2.0   # 1.37
SINGLES_RIGHT = SINGLES_LEFT + SINGLES_WIDTH           # 9.60
CENTER_X = DOUBLES_WIDTH / 2.0
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

COURT_TYPES = ("doubles", "singles")


# ── Camera ─────────────────────────────────────────────────────────────────
@dataclass
class CameraModel:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    focal_px: float
    confidence: float

    def ray(self, image_point: Sequence[float]) -> np.ndarray:
        """Direzione unitaria nel mondo che parte dalla camera e passa per il pixel."""
        p = np.array([image_point[0], image_point[1], 1.0], dtype=np.float64)
        d_cam = np.linalg.inv(self.K) @ p
        d_world = self.R.T @ d_cam
        norm = np.linalg.norm(d_world)
        if norm < 1e-12:
            raise ValueError("Raggio non valido")
        return d_world / norm


def _focal_from_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    """Stima la focale (pixel) da una omografia del piano, assumendo:
    pixel quadrati, skew nullo, centro ottico noto (cx, cy).
    """
    h1, h2 = H[:, 0], H[:, 1]

    def v(hi, hj):
        return np.array(
            [
                hi[0] * hj[0],
                hi[0] * hj[1] + hi[1] * hj[0],
                hi[1] * hj[1],
                hi[2] * hj[0] + hi[0] * hj[2],
                hi[2] * hj[1] + hi[1] * hj[2],
                hi[2] * hj[2],
            ],
            dtype=np.float64,
        )

    v12 = v(h1, h2)
    v11 = v(h1, h1)
    v22 = v(h2, h2)
    a = np.array([1.0, 0.0, 1.0, -cx, -cy, cx * cx + cy * cy], dtype=np.float64)
    d1 = float(v12 @ a)
    n1 = float(v12[5])
    d2 = float((v11 - v22) @ a)
    n2 = float((v11 - v22)[5])
    den = d1 * d1 + d2 * d2
    if den < 1e-12:
        return None
    w = -(d1 * n1 + d2 * n2) / den
    if w <= 1e-12:
        return None
    f = 1.0 / math.sqrt(w)
    if not math.isfinite(f):
        return None
    return f


def _rodrigues_orthonormalize(r1: np.ndarray, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 - (r1 @ r2) * r1
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    return r1, r2, r3


def _decompose_once(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1 = lam * M[:, 0]
    r2 = lam * M[:, 1]
    t = lam * M[:, 2]
    r1, r2, r3 = _rodrigues_orthonormalize(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    C = -R.T @ t
    ortho = abs(float(r1 @ r2))
    return R, t, C, ortho


# ── Modello campo ──────────────────────────────────────────────────────────
class CourtModel:
    """Campo da tennis calibrato da 4 angoli cliccati sull'immagine.

    Ordine dei 4 punti, come appaiono nell'immagine:
        alto-sinistra, alto-destra, basso-destra, basso-sinistra

    Convenzione coordinate campo:
        X = 0 .. 10.97  (larghezza doppio, da sinistra a destra)
        Y = 0 .. 23.77  (0 = base vicina in basso, 23.77 = base lontana in alto)
        La rete è a Y = 11.885.

    Se `court_type="singles"` i 4 punti sono le righe del singolo;
    l'omografia viene comunque riportata nel sistema doppio (X 1.37..9.60).
    """

    def __init__(
        self,
        image_corners: Sequence[Sequence[float]],
        court_type: str = "doubles",
        image_size: Sequence[int] | None = None,
    ):
        pts = np.asarray(image_corners, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] != 4:
            raise ValueError("Servono esattamente 4 angoli del campo")
        if court_type not in COURT_TYPES:
            raise ValueError(f"court_type deve essere uno di {COURT_TYPES}")
        self.image_corners = pts
        self.court_type = court_type
        self.image_size = tuple(image_size) if image_size is not None else None
        self.court_corners = self._court_corners(court_type)

        H, _ = cv2.findHomography(self.image_corners, self.court_corners)
        if H is None:
            raise ValueError("Impossibile calcolare l'omografia: angoli degeneri?")
        self.H = H  # immagine -> campo
        self.H_inv = np.linalg.inv(H)
        self.camera = self._calibrate_camera()

    # ── Costruzione ────────────────────────────────────────────────────────
    @staticmethod
    def _court_corners(court_type: str) -> np.ndarray:
        if court_type == "doubles":
            x0, x1 = 0.0, DOUBLES_WIDTH
        else:
            x0, x1 = SINGLES_LEFT, SINGLES_RIGHT
        # ordine: alto-sx, alto-dx, basso-dx, basso-sx (vista immagine)
        return np.array(
            [
                [x0, COURT_LENGTH],
                [x1, COURT_LENGTH],
                [x1, 0.0],
                [x0, 0.0],
            ],
            dtype=np.float64,
        )

    def _calibrate_camera(self) -> CameraModel | None:
        if self.image_size is None:
            # stima grossolana dal bounding box degli angoli
            w = max(self.image_corners[:, 0]) - min(self.image_corners[:, 0])
            h = max(self.image_corners[:, 1]) - min(self.image_corners[:, 1])
            self.image_size = (int(round(w)), int(round(h)))
        cx = self.image_size[0] / 2.0
        cy = self.image_size[1] / 2.0

        # La decomposizione camera richiede H campo -> immagine.
        # self.H è immagine -> campo, quindi usiamo la sua inversa.
        H_cam = self.H_inv
        scale = abs(H_cam[2, 2])
        Hn = H_cam / (H_cam[2, 2] if scale > 1e-12 else np.linalg.norm(H_cam))
        f = _focal_from_homography(Hn, cx, cy)
        if f is None:
            return None
        if f < 0.15 * max(self.image_size) or f > 30.0 * max(self.image_size):
            return None

        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        best = None
        best_score = -1.0
        # L'omografia è definita a meno di segno: prova entrambe le versioni.
        for sign in (1.0, -1.0):
            R, t, C, ortho = _decompose_once(sign * Hn, K)
            if C[2] <= 0:
                continue
            # Più il centro camera è sopra il campo e la rotazione è ortogonale,
            # più la decomposizione è plausibile.
            score = 1.0 / (1e-6 + ortho) + min(C[2], 30.0)
            if score > best_score:
                best_score = score
                best = (R, t, C, ortho)

        if best is None:
            return None

        R, t, C, ortho = best
        # Confidenza: ortogonalità + altezza camera plausibile + focale plausibile
        conf = 1.0 / (1.0 + 50.0 * ortho)
        if not (2.0 <= C[2] <= 30.0):
            conf *= 0.5
        return CameraModel(K=K, R=R, t=t, C=C, focal_px=f, confidence=float(conf))

    # ── Conversioni ────────────────────────────────────────────────────────
    def image_to_court(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return out.astype(np.float64)

    def court_to_image(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)
        return out.astype(np.float64)

    def distance_m(self, p1: Sequence[float], p2: Sequence[float]) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        return float(np.linalg.norm(a - b))

    # ── Regole campo ───────────────────────────────────────────────────────
    def is_in_court(
        self,
        point: Sequence[float],
        mode: str | None = None,
        tolerance_m: float = 0.02,
    ) -> bool:
        """True se il punto (coordinate campo) è dentro l'area richiesta.

        mode: "singles", "doubles", "service_near", "service_far".
        Default: il tipo di campo con cui è stato calibrato.
        """
        if mode is None:
            mode = self.court_type
        x, y = float(point[0]), float(point[1])
        tol = tolerance_m

        if mode in ("singles", "doubles"):
            if mode == "doubles":
                x0, x1 = 0.0, DOUBLES_WIDTH
            else:
                x0, x1 = SINGLES_LEFT, SINGLES_RIGHT
            return (x0 - tol) <= x <= (x1 + tol) and (-tol) <= y <= (COURT_LENGTH + tol)

        if mode in ("service_near", "service_far"):
            x0, x1 = SINGLES_LEFT, SINGLES_RIGHT
            if mode == "service_near":
                y0, y1 = NET_Y - SERVICE_LINE_OFFSET, NET_Y
            else:
                y0, y1 = NET_Y, NET_Y + SERVICE_LINE_OFFSET
            return (x0 - tol) <= x <= (x1 + tol) and (y0 - tol) <= y <= (y1 + tol)

        raise ValueError(f"mode non valido: {mode}")

    def nearest_line_distance_m(self, point: Sequence[float]) -> float:
        """Distanza dalla riga più vicina del campo doppio (utile per 'vincente')."""
        x, y = float(point[0]), float(point[1])
        xs = [0.0, SINGLES_LEFT, SINGLES_RIGHT, DOUBLES_WIDTH, CENTER_X]
        ys = [0.0, NET_Y - SERVICE_LINE_OFFSET, NET_Y, NET_Y + SERVICE_LINE_OFFSET, COURT_LENGTH]
        return min(min(abs(x - v) for v in xs), min(abs(y - v) for v in ys))

    # ── Velocità ───────────────────────────────────────────────────────────
    @staticmethod
    def speed_kmh(distance_m: float, dt_s: float) -> float:
        if dt_s <= 1e-9:
            return 0.0
        return (distance_m / dt_s) * 3.6

    def bounce_speed_kmh(
        self,
        bounce1_court: Sequence[float],
        bounce2_court: Sequence[float],
        dt_s: float,
    ) -> float:
        """Velocità media orizzontale tra due rimbalzi.

        È fisicamente corretta perché a entrambi i rimbalzi la palla è a
        quota zero, quindi l'omografia dà la posizione reale.
        """
        return self.speed_kmh(self.distance_m(bounce1_court, bounce2_court), dt_s)

    # ── Fit balistico 3D ───────────────────────────────────────────────────
    def fit_ballistic(
        self,
        image_points: Sequence[Sequence[float]],
        timestamps_s: Sequence[float],
        robust: bool = True,
    ) -> "BallisticFit | None":
        """Stima la traiettoria 3D della palla da una sequenza di pixel.

        Modello: X(t) = X0 + V0*t + 0.5*g*t^2.
        Vincolo: X(t) deve giacere sul raggio camera-pixel.
        Le equazioni sono lineari in X0 e V0 (prodotto vettoriale), quindi
        si risolve con least squares; se `robust`, ripesa gli outlier (IRLS).

        Richiede `self.camera`. Ritorna None se la calibrazione non è
        disponibile o se ci sono meno di 5 punti.
        """
        if self.camera is None:
            return None
        pts = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        ts = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
        if len(pts) != len(ts) or len(pts) < 5:
            return None
        if len(pts) < 5:
            return None

        C = self.camera.C
        K_inv = np.linalg.inv(self.camera.K)
        R = self.camera.R

        A_rows = []
        b_rows = []
        dirs = []
        for (u, v), ti in zip(pts, ts):
            d_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
            d = R.T @ d_cam
            d /= np.linalg.norm(d)
            dirs.append(d)
            D = np.array(
                [[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]],
                dtype=np.float64,
            )
            A_rows.append(np.hstack([D, ti * D]))
            b_rows.append(D @ C - 0.5 * ti * ti * (D @ GRAVITY))

        A = np.vstack(A_rows)
        b = np.hstack(b_rows)
        weights = np.ones(len(pts), dtype=np.float64)

        sol = None
        for _ in range(5 if robust else 1):
            w = np.repeat(weights, 3)
            Aw = A * w[:, None]
            bw = b * w
            sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            X0, V0 = sol[:3], sol[3:]
            X = X0[None, :] + ts[:, None] * V0[None, :] + 0.5 * ts[:, None] ** 2 * GRAVITY[None, :]
            residual = np.linalg.norm(np.cross(X - C, np.asarray(dirs)), axis=1)
            med = float(np.median(residual)) + 1e-6
            weights = 1.0 / (1.0 + (residual / (3.0 * med)) ** 2)
            weights = np.clip(weights, 0.05, 1.0)

        if sol is None:
            return None
        X0, V0 = sol[:3], sol[3:]
        X = X0[None, :] + ts[:, None] * V0[None, :] + 0.5 * ts[:, None] ** 2 * GRAVITY[None, :]
        V = V0[None, :] + ts[:, None] * GRAVITY[None, :]
        residual = np.linalg.norm(np.cross(X - C, np.asarray(dirs)), axis=1)
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        speed = np.linalg.norm(V, axis=1) * 3.6
        conf = 1.0 / (1.0 + rmse) * min(1.0, len(pts) / 15.0)
        return BallisticFit(
            X0=X0,
            V0=V0,
            timestamps=ts,
            positions=X,
            velocities=V,
            speed_kmh=speed,
            rmse_m=rmse,
            confidence=float(conf),
        )

    # ── Serializzazione ────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        data = {
            "court": self.court_type,
            "polygon": self.image_corners.tolist(),
            "image_corners": self.image_corners.tolist(),
            "court_corners": self.court_corners.tolist(),
            "homography": self.H.tolist(),
            "homography_inv": self.H_inv.tolist(),
            "court_dimensions": {
                "length_m": COURT_LENGTH,
                "doubles_width_m": DOUBLES_WIDTH,
                "singles_width_m": SINGLES_WIDTH,
                "net_y_m": NET_Y,
            },
        }
        if self.image_size is not None:
            data["frame_size"] = [int(self.image_size[0]), int(self.image_size[1])]
        if self.camera is not None:
            data["camera"] = {
                "focal_px": float(self.camera.focal_px),
                "K": self.camera.K.tolist(),
                "R": self.camera.R.tolist(),
                "t": self.camera.t.tolist(),
                "center_m": self.camera.C.tolist(),
                "confidence": float(self.camera.confidence),
            }
        return data

    @classmethod
    def from_roi_json(cls, path: str | Path) -> "CourtModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # polygon/image_corners sono i 4 punti in pixel; court_corners sono
        # le coordinate in metri e NON vanno usate come punti immagine.
        corners = data.get("image_corners") or data.get("polygon")
        if not corners:
            raise ValueError(f"Nessun poligono/immagine in {path}")
        court_type = data.get("court", "doubles")
        image_size = data.get("frame_size")
        return cls(corners, court_type=court_type, image_size=image_size)


@dataclass
class BallisticFit:
    X0: np.ndarray
    V0: np.ndarray
    timestamps: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    speed_kmh: np.ndarray
    rmse_m: float
    confidence: float

    def speed_at(self, t: float) -> float:
        v = self.V0 + np.asarray([0.0, 0.0, -9.81]) * t
        return float(np.linalg.norm(v) * 3.6)


# ── Utility per tracce ──────────────────────────────────────────────────────
def speed_between_bounces(
    model: CourtModel,
    bounce1_court: Sequence[float],
    bounce2_court: Sequence[float],
    frame1: int,
    frame2: int,
    fps: float,
) -> dict:
    dt = abs(frame2 - frame1) / fps
    dist = model.distance_m(bounce1_court, bounce2_court)
    kmh = model.speed_kmh(dist, dt)
    return {
        "from_frame": int(frame1),
        "to_frame": int(frame2),
        "distance_m": round(dist, 3),
        "dt_s": round(dt, 4),
        "speed_kmh": round(kmh, 1),
        "valid": bool(0.05 <= dt <= 3.0 and dist <= 35.0),
    }

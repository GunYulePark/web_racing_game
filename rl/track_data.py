from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TrackSpec:
    name: str
    points: np.ndarray


def make_procedural_track(n: int, cx: float, cz: float, sx: float, sz: float, fn: Callable[[float], float]) -> np.ndarray:
    pts = []
    for i in range(n):
        a = (i / n) * math.pi * 2.0
        r = fn(a)
        pts.append((cx + math.cos(a) * r * sx, cz + math.sin(a) * r * sz))
    return np.asarray(pts, dtype=np.float32)


def build_tracks() -> dict[str, TrackSpec]:
    classic = np.asarray([
        (-320, -40), (-250, -190), (-40, -250), (180, -200), (315, -70), (320, 70),
        (210, 145), (45, 155), (-60, 105), (10, 20), (180, 40), (260, 190),
        (120, 300), (-90, 315), (-280, 260), (-355, 120),
    ], dtype=np.float32)

    stadium = make_procedural_track(
        40, -120, -20, 1.12, 0.78,
        lambda a: (520 + 170 * math.cos(a) - 110 * math.cos(2 * a) + 55 * math.sin(3 * a)
                   + (130 if 1.1 < a < 2.05 else 0)
                   - (90 if 4.2 < a < 5.0 else 0))
    )
    coastal = make_procedural_track(
        42, 0, 20, 1.08, 0.82,
        lambda a: 500 + 135 * math.sin(a + 0.35) + 95 * math.cos(2.6 * a) + 40 * math.sin(4.2 * a),
    )
    canyon = make_procedural_track(
        38, -40, 0, 1.0, 0.86,
        lambda a: 470 + 190 * math.cos(a * 1.1) - 75 * math.sin(a * 2.8) - (120 if 2.0 < a < 3.15 else 0),
    )
    nightcity = make_procedural_track(
        44, -20, -10, 1.05, 0.8,
        lambda a: 510 + 110 * math.cos(3 * a) + 80 * math.sin(2 * a + 0.8),
    )
    return {
        'classic': TrackSpec('classic', classic),
        'stadium': TrackSpec('stadium', stadium),
        'coastal': TrackSpec('coastal', coastal),
        'canyon': TrackSpec('canyon', canyon),
        'nightcity': TrackSpec('nightcity', nightcity),
    }

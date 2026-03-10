from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackSample:
    t: float
    point: np.ndarray
    tangent: np.ndarray
    right: np.ndarray
    signed_offset: float
    distance: float


class TrackModel:
    def __init__(self, points: np.ndarray):
        if len(points) < 3:
            raise ValueError('track needs at least 3 points')
        self.points = np.asarray(points, dtype=np.float32)
        self.count = len(self.points)
        self.segment_lengths = np.linalg.norm(np.roll(self.points, -1, axis=0) - self.points, axis=1)
        self.track_length = float(np.sum(self.segment_lengths))

    def wrap_t(self, t: float) -> float:
        return t % 1.0

    def frame(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tf = self.wrap_t(t)
        idx_f = tf * self.count
        i0 = int(math.floor(idx_f)) % self.count
        i1 = (i0 + 1) % self.count
        alpha = idx_f - math.floor(idx_f)
        p0 = self.points[i0]
        p1 = self.points[i1]
        point = p0 * (1.0 - alpha) + p1 * alpha
        tangent = p1 - p0
        tangent /= np.linalg.norm(tangent) + 1e-8
        right = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        return point.astype(np.float32), tangent.astype(np.float32), right

    def nearest(self, x: float, z: float, *, samples: int = 300, around_t: float | None = None, window: float = 0.12, refine_passes: int = 2) -> TrackSample:
        best_t = 0.0
        best_point, best_tangent, best_right = self.frame(0.0)
        best_dist_sq = float('inf')

        if around_t is None:
            start = 0.0
            end = 1.0
        else:
            start = around_t - window
            end = around_t + window

        def search(range_start: float, range_end: float, count: int) -> None:
            nonlocal best_t, best_point, best_tangent, best_right, best_dist_sq
            for i in range(max(2, count) + 1):
                alpha = i / max(1, count)
                t = range_start + (range_end - range_start) * alpha
                point, tangent, right = self.frame(t)
                dx = x - point[0]
                dz = z - point[1]
                dist_sq = dx * dx + dz * dz
                if dist_sq < best_dist_sq:
                    best_t = self.wrap_t(t)
                    best_point = point
                    best_tangent = tangent
                    best_right = right
                    best_dist_sq = dist_sq

        search(start, end, samples)
        radius = window * 0.45 if around_t is not None else 0.08
        for _ in range(refine_passes):
            search(best_t - radius, best_t + radius, max(24, int(samples * 0.35)))
            radius *= 0.35

        dx = x - best_point[0]
        dz = z - best_point[1]
        signed_offset = float(dx * best_right[0] + dz * best_right[1])
        return TrackSample(
            t=best_t,
            point=best_point,
            tangent=best_tangent,
            right=best_right,
            signed_offset=signed_offset,
            distance=math.sqrt(best_dist_sq),
        )

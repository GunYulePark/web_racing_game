from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def heuristic_action(obs: np.ndarray) -> np.ndarray:
    speed = float(obs[0])
    heading_err = float(obs[7]) * np.pi
    slip_angle = float(obs[8]) * 1.2
    curvature_near = float(obs[10])
    curvature_far = float(obs[12])
    offroad_margin = float(obs[9])

    corner_need = abs(curvature_near) * 0.9 + abs(curvature_far) * 1.1 + abs(heading_err) * 0.55
    steer = np.clip(-(heading_err * 1.45 + slip_angle * 0.7), -1.0, 1.0)
    throttle = np.clip(0.96 - corner_need * 0.8 - max(0.0, offroad_margin) * 0.45 - max(0.0, speed - 0.88) * 0.9, 0.14, 1.0)
    brake = np.clip(corner_need * 0.75 + max(0.0, speed - 0.82) * 0.85 + max(0.0, offroad_margin) * 0.5, 0.0, 1.0)
    drift = np.clip(corner_need * 0.85 - max(0.0, offroad_margin) * 1.4, 0.0, 1.0)
    return np.asarray([steer, throttle, brake, drift], dtype=np.float32)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


class DenseJsonPolicy:
    def __init__(self, payload: dict):
        self.payload = payload
        self.layers = payload.get('layers', [])
        if not self.layers:
            raise ValueError('policy JSON has no layers')

    @classmethod
    def from_path(cls, path: str | Path) -> 'DenseJsonPolicy':
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(payload)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for layer in self.layers:
            weights = np.asarray(layer['weights'], dtype=np.float32)
            bias = np.asarray(layer['bias'], dtype=np.float32)
            x = weights @ x + bias
            if layer.get('activation') == 'relu':
                x = _relu(x)
        out = np.asarray(x, dtype=np.float32)
        out[0] = np.clip(out[0], -1.0, 1.0)
        out[1:] = np.clip(out[1:], 0.0, 1.0)
        return out

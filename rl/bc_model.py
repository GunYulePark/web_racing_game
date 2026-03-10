from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch
from torch import nn


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int = 20, action_dim: int = 4, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            last_dim = hidden
        layers.append(nn.Linear(last_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.model(obs)
        steer = torch.tanh(raw[..., 0:1])
        pedals = torch.sigmoid(raw[..., 1:4])
        return torch.cat([steer, pedals], dim=-1)


@dataclass
class ExportLayer:
    activation: str
    weights: list[list[float]]
    bias: list[float]


def export_policy_json(model: BCPolicy, out_path: str | Path, *, name: str = 'bc-policy') -> dict:
    dense_layers = [module for module in model.model if isinstance(module, nn.Linear)]
    payload = {
        'format': 'dense-mlp-v1',
        'name': name,
        'observationDim': dense_layers[0].in_features,
        'actionDim': dense_layers[-1].out_features,
        'layers': [],
    }
    for index, layer in enumerate(dense_layers):
        payload['layers'].append({
            'activation': 'relu' if index < len(dense_layers) - 1 else 'linear',
            'weights': layer.weight.detach().cpu().tolist(),
            'bias': layer.bias.detach().cpu().tolist(),
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload

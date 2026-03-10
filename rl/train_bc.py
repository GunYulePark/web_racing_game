from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bc_model import BCPolicy


def load_dataset(path: Path):
    bundle = np.load(path)
    obs = bundle['obs'].astype(np.float32)
    actions = bundle['actions'].astype(np.float32)
    weights = bundle['sample_weight'].astype(np.float32)
    train_idx = bundle['train_idx']
    val_idx = bundle['val_idx']
    obs_mean = bundle['obs_mean'].astype(np.float32)
    obs_std = bundle['obs_std'].astype(np.float32)
    return obs, actions, weights, train_idx, val_idx, obs_mean, obs_std


def normalize(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (obs - mean) / std


def make_loader(obs: np.ndarray, actions: np.ndarray, weights: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(obs[idx]),
        torch.from_numpy(actions[idx]),
        torch.from_numpy(weights[idx]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for obs, actions, weights in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            weights = weights.to(device)
            pred = model(obs)
            loss = ((pred - actions) ** 2).mean(dim=1) * weights
            loss_sum += float(loss.sum().item())
            count += int(loss.shape[0])
    return loss_sum / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a small behavior-cloning policy from telemetry dataset')
    parser.add_argument('dataset', help='Path to bc_dataset.npz')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else dataset_path.parent / 'bc_runs'
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, actions, weights, train_idx, val_idx, obs_mean, obs_std = load_dataset(dataset_path)
    obs = normalize(obs, obs_mean, obs_std)

    train_loader = make_loader(obs, actions, weights, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(obs, actions, weights, val_idx, args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BCPolicy(obs_dim=obs.shape[1], action_dim=actions.shape[1], hidden_sizes=tuple(args.hidden)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_val = float('inf')
    best_path = out_dir / 'bc_policy.pt'

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_obs, batch_actions, batch_weights in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_weights = batch_weights.to(device)
            pred = model(batch_obs)
            loss = ((pred - batch_actions) ** 2).mean(dim=1) * batch_weights
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = evaluate(model, train_loader, device)
        val_loss = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(json.dumps(history[-1]))
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'obs_mean': obs_mean,
                'obs_std': obs_std,
                'hidden_sizes': args.hidden,
                'obs_dim': int(obs.shape[1]),
                'action_dim': int(actions.shape[1]),
            }, best_path)

    (out_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps({'checkpoint': str(best_path), 'best_val_loss': best_val}, indent=2))


if __name__ == '__main__':
    main()

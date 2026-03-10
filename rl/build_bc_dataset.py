from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Build behavior-cloning dataset from telemetry step JSONL')
    parser.add_argument('input', help='Path to *.steps.jsonl generated from telemetry_to_jsonl.mjs')
    parser.add_argument('--output-dir', default=None, help='Directory for dataset outputs (default: alongside input)')
    parser.add_argument('--min-speed', type=float, default=4.0, help='Filter out near-stationary samples')
    parser.add_argument('--max-offroad-margin', type=float, default=1.75, help='Filter extreme recovery samples')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent / f'{input_path.stem}_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    filtered = [
        row for row in rows
        if len(row.get('obs', [])) == 20
        and len(row.get('action', [])) == 4
        and float(row.get('speed', 0.0)) >= args.min_speed
        and float(row.get('offroadMargin', 0.0)) <= args.max_offroad_margin
    ]
    if not filtered:
        raise SystemExit('No usable samples after filtering')

    obs = np.asarray([row['obs'] for row in filtered], dtype=np.float32)
    actions = np.asarray([row['action'] for row in filtered], dtype=np.float32)
    weights = np.asarray([
        1.0
        + min(2.5, max(0.0, float(row.get('reward', 0.0))) * 0.08)
        + min(1.5, float(row.get('stability', 0.0)) * 0.4)
        - min(0.75, float(row.get('offroad', 0)) * 0.4)
        - min(0.75, float(row.get('contact', 0)) * 0.5)
        for row in filtered
    ], dtype=np.float32)
    weights = np.clip(weights, 0.2, 4.0)

    rng = np.random.default_rng(12345)
    indices = np.arange(len(filtered))
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * args.val_ratio))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    if len(train_idx) == 0:
        train_idx = val_idx
        val_idx = indices[:1]

    obs_mean = obs[train_idx].mean(axis=0)
    obs_std = obs[train_idx].std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)

    np.savez_compressed(
        out_dir / 'bc_dataset.npz',
        obs=obs,
        actions=actions,
        sample_weight=weights,
        train_idx=train_idx,
        val_idx=val_idx,
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
    )

    manifest = {
        'input': str(input_path),
        'samples_total': int(len(filtered)),
        'samples_filtered_from': int(len(rows)),
        'train_samples': int(len(train_idx)),
        'val_samples': int(len(val_idx)),
        'obs_dim': 20,
        'action_dim': 4,
        'min_speed': args.min_speed,
        'max_offroad_margin': args.max_offroad_margin,
        'drivers': sorted({row.get('driver', 'unknown') for row in filtered}),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({
        'dataset': str(out_dir / 'bc_dataset.npz'),
        'manifest': str(out_dir / 'manifest.json'),
        **manifest,
    }, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bc_model import BCPolicy, export_policy_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Export a trained BC checkpoint to browser-friendly JSON')
    parser.add_argument('checkpoint', help='Path to bc_policy.pt')
    parser.add_argument('--output', default=None, help='Path to exported JSON')
    parser.add_argument('--name', default='bc-policy')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location='cpu')
    hidden_sizes = tuple(payload.get('hidden_sizes', [128, 128]))
    model = BCPolicy(
        obs_dim=int(payload.get('obs_dim', 20)),
        action_dim=int(payload.get('action_dim', 4)),
        hidden_sizes=hidden_sizes,
    )
    model.load_state_dict(payload['state_dict'])
    output_path = Path(args.output).resolve() if args.output else checkpoint_path.with_suffix('.policy.json')
    exported = export_policy_json(model, output_path, name=args.name)
    print({'output': str(output_path), 'layers': len(exported['layers'])})


if __name__ == '__main__':
    main()

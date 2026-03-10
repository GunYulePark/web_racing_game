from __future__ import annotations

import argparse
import json
from statistics import mean

import numpy as np

from policies import DenseJsonPolicy, heuristic_action
from racing_env import RacingEnv


def run_episode(env: RacingEnv, mode: str, seed: int | None = None, policy: DenseJsonPolicy | None = None) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        if mode == 'heuristic':
            action = heuristic_action(obs)
        elif mode == 'json-policy':
            if policy is None:
                raise ValueError('json-policy mode requires a loaded policy')
            action = policy.predict(obs)
        else:
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            return {
                "reward": total_reward,
                "steps": steps,
                "track_progress": info["track_progress"],
                "lap_count": info["lap_count"],
                "offroad": info["offroad"],
                "stability_proxy": max(0.0, 1.0 - abs(info["heading_error"]) / np.pi - info["slip_angle"] / 1.8),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick evaluation helper for the racing RL env")
    parser.add_argument("--track", default="stadium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--random", action="store_true", help="use random actions instead of the built-in heuristic")
    parser.add_argument("--policy-json", default=None, help="optional exported dense policy JSON to evaluate")
    args = parser.parse_args()

    mode = 'json-policy' if args.policy_json else ('random' if args.random else 'heuristic')
    policy = DenseJsonPolicy.from_path(args.policy_json) if args.policy_json else None

    env = RacingEnv(track=args.track)
    results = [run_episode(env, mode=mode, seed=i, policy=policy) for i in range(args.episodes)]

    summary = {
        "track": args.track,
        "episodes": args.episodes,
        "policy": mode,
        "mean_reward": mean(item["reward"] for item in results),
        "mean_steps": mean(item["steps"] for item in results),
        "mean_progress": mean(item["track_progress"] for item in results),
        "mean_stability_proxy": mean(item["stability_proxy"] for item in results),
        "laps_finished": sum(1 for item in results if item["lap_count"] >= 1),
        "raw": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

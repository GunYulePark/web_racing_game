from __future__ import annotations

import argparse
import json
from statistics import mean

import numpy as np

from racing_env import RacingEnv


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


def run_episode(env: RacingEnv, use_heuristic: bool, seed: int | None = None) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = heuristic_action(obs) if use_heuristic else env.action_space.sample()
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
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick evaluation helper for the racing RL env")
    parser.add_argument("--track", default="stadium")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--random", action="store_true", help="use random actions instead of the built-in heuristic")
    args = parser.parse_args()

    env = RacingEnv(track=args.track)
    results = [run_episode(env, use_heuristic=not args.random, seed=i) for i in range(args.episodes)]

    summary = {
        "track": args.track,
        "episodes": args.episodes,
        "policy": "random" if args.random else "heuristic",
        "mean_reward": mean(item["reward"] for item in results),
        "mean_steps": mean(item["steps"] for item in results),
        "mean_progress": mean(item["track_progress"] for item in results),
        "laps_finished": sum(1 for item in results if item["lap_count"] >= 1),
        "raw": results,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

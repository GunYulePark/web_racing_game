# Racing RL Starter

This folder is the first real step from scripted racing AI to reinforcement learning.

## What is included

- `track_data.py` — track definitions matching the browser game's layout style
- `track_geometry.py` — reusable projection / nearest-point helpers for closed racing lines
- `racing_env.py` — Gymnasium-compatible continuous-control racing environment
- `train_ppo.py` — PPO training entrypoint using Stable-Baselines3
- `evaluate_env.py` — quick heuristic-vs-random sanity check runner
- `test_env.py` — lightweight regression checks for geometry + env API
- `requirements.txt` — Python dependencies

## Observation space

Current observation vector contains:

- normalized speed
- lateral speed
- yaw rate
- steer
- throttle
- brake
- signed distance from track center
- heading error vs track tangent
- slip angle
- off-road margin
- future curvature hints (5 lookaheads)
- heading sin/cos
- normalized track position
- normalized episode progress
- off-road flag

## Action space

Continuous 4D action:

1. steering `[-1, 1]`
2. throttle `[0, 1]`
3. brake `[0, 1]`
4. drift bias `[0, 1]`

## Reward design

The reward currently favors:

- forward track progress
- stable speed
- staying close to the racing line
- avoiding off-road and spin states
- avoiding unnecessary braking and unstable steering
- not drifting on easy straight sections

## Quick start

### 1. Install deps

```bash
cd web_racing_game/rl
pip install -r requirements.txt
```

### 2. Train PPO

```bash
python train_ppo.py
```

### 3. Run a quick evaluation sanity check

```bash
python evaluate_env.py --track stadium --episodes 5
```

### 4. Evaluate manually

Open a Python shell and run:

```python
from racing_env import RacingEnv
env = RacingEnv(track="stadium")
obs, info = env.reset()
for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(reward, info)
    if terminated or truncated:
        break
```

## Recommended next steps

1. Match the browser physics more closely
   - port more of the JS handling model into the Python env
   - align grip / drag / yaw constants

2. Add opponent cars
   - start with one blocking opponent
   - later add slipstream and overtaking reward shaping

3. Export trained policy back to web
   - easiest path: tiny MLP weights → JSON → manual JS inference
   - alternative: convert to ONNX or TensorFlow.js

4. Add imitation-learning bootstrap
   - use your browser telemetry as demonstrations
   - pretrain policy before PPO fine-tuning
   - browser export helper: `window.downloadRacingAITelemetry()`
   - offline conversion helper: `node ../tools/telemetry_to_jsonl.mjs ../telemetry.json`

## Important note

This is a **training scaffold**, not a perfect physics mirror of the browser version yet.
That is intentional: get the RL loop working first, then close the sim-to-game gap.

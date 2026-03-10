# Racing AI RL Notes

## Current foundation

The game now includes lightweight telemetry for rival drivers:

- per-lap summaries
- per-segment reward accumulation
- contact count
- drift usage
- off-road frequency

At runtime you can inspect/export it from the browser console:

- `window.getRacingAITelemetry()`
- `window.dumpRacingAITelemetry()`
- `window.downloadRacingAITelemetry()`

New segment telemetry now also captures:

- throttle/brake usage
- progress delta
- stability loss / recovery pressure

Telemetry is also stored in localStorage under:

- `racingAIDebugTelemetry`

Offline conversion helper:

- `node tools/telemetry_to_jsonl.mjs ./telemetry.json`
- writes `*.episodes.jsonl` and `*.segments.jsonl` for imitation/offline RL analysis

## Recommended RL path

### Phase 1 — Imitation / behavior cloning

Train a policy from good scripted laps first.

State candidates:
- speed forward
- lateral velocity
- yaw rate
- steering angle
- signed distance from racing line
- upcoming curve samples (near/mid/far)
- opponent distance ahead/side
- tire wear / damage

Action candidates:
- steer in [-1, 1]
- throttle in [0, 1]
- brake in [0, 1]
- optional drift bias in [0, 1]

Reward:
- +track progress
- +lap completion bonus
- -off-road
- -collision
- -spin / large heading error
- -unnecessary brake on straight
- small -control effort penalty

### Phase 2 — Offline RL / simulator training

Run many episodes outside the browser and train with:

- PPO for stable continuous control
- SAC if you want stronger continuous action exploration

Best first choice here: **PPO**

Why:
- easier to tune
- robust on noisy racing tasks
- many reference implementations

### Phase 3 — Distillation back into game

Export trained policy weights.
Possible serving options:

1. TensorFlow.js model directly in browser
2. tiny MLP weights exported to JSON and evaluated manually in JS
3. ONNX model via web runtime

For this game, option 2 or 1 is the simplest.

## Practical architecture

### Option A — Keep browser as renderer only

- port physics to Node/Python simulator
- run 1000s of episodes headlessly
- train with Python stack (stable-baselines3 / CleanRL / Ray RLlib)
- export policy back to JS

This is the most realistic route.

### Option B — Train in browser

Possible, but slower and much more awkward.
Only worth it for demos.

## Minimal reward formula

A good starting reward per step:

`reward = progressDelta * 10 - offRoad * 3 - collision * 4 - spinPenalty * 2 - steerJerk * 0.03 - brakeOnStraight * 0.02`

Then add:

- `+50` on lap completion
- `-20` for large crash / stuck condition

## First real milestone

Before true RL, build a headless env with:

- reset()
- step(action)
- observation vector
- reward
- done

Gym-style shape:

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

Once that exists, PPO becomes straightforward.

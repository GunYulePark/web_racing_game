from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from racing_env import RacingEnv


def make_env(track: str = "stadium"):
    def _factory():
        return Monitor(RacingEnv(track=track))
    return _factory


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "runs" / "ppo_stadium"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env("stadium"), n_envs=8)
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        tensorboard_log=str(out_dir / "tb"),
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True)
    model.save(out_dir / "ppo_racing_agent")
    print(f"saved model to {out_dir}")

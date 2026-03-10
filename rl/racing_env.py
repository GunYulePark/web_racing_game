from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from track_data import build_tracks
from track_geometry import TrackModel


@dataclass
class CarState:
    x: float = 0.0
    z: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    lateral_speed: float = 0.0
    yaw_rate: float = 0.0
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, track: str = "stadium", dt: float = 0.1, max_steps: int = 3000):
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.road_half_width = 20.0
        self.max_speed = 78.0
        self.max_reverse = -8.0
        self.accel_rate = 18.0
        self.brake_rate = 24.0
        self.drag_linear = 0.55
        self.drag_quad = 0.0038
        self.base_grip = 3.2
        self.corner_stiffness = 0.24
        self.tracks = build_tracks()
        self.track_name = track if track in self.tracks else "stadium"
        self.track_points = self.tracks[self.track_name].points
        self.track = TrackModel(self.track_points)
        self.state = CarState()
        self.step_count = 0
        self.progress = 0.0
        self.prev_progress = 0.0
        self.lap_count = 0
        self.last_track_t = 0.0
        self.visited_segments: set[int] = set()
        self.last_info: dict[str, Any] = {}

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(20,), dtype=np.float32)

    def _wrap_angle(self, a: float) -> float:
        while a > math.pi:
            a -= math.pi * 2.0
        while a < -math.pi:
            a += math.pi * 2.0
        return a

    def _track_frame(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.track.frame(t)

    def _nearest(self, x: float, z: float, samples: int = 300, around_t: float | None = None):
        return self.track.nearest(x, z, samples=samples, around_t=around_t)

    def _obs(self) -> np.ndarray:
        nearest = self._nearest(self.state.x, self.state.z, around_t=self.last_track_t)
        t = nearest.t
        tangent = nearest.tangent
        heading_track = math.atan2(tangent[0], tangent[1])
        heading_err = self._wrap_angle(self.state.heading - heading_track)
        offroad_margin = (abs(nearest.signed_offset) - self.road_half_width) / self.road_half_width
        slip_angle = math.atan2(self.state.lateral_speed, max(1.0, abs(self.state.speed)))

        future = []
        for delta in (0.01, 0.025, 0.05, 0.085, 0.12):
            _, tangent_f, _ = self._track_frame(t + delta)
            future_heading = math.atan2(tangent_f[0], tangent_f[1])
            future.append(self._wrap_angle(future_heading - heading_track))

        obs = np.array([
            self.state.speed / self.max_speed,
            self.state.lateral_speed / 30.0,
            self.state.yaw_rate / 2.5,
            self.state.steer,
            self.state.throttle,
            self.state.brake,
            nearest.signed_offset / self.road_half_width,
            heading_err / math.pi,
            slip_angle / 1.2,
            offroad_margin,
            *future,
            math.sin(self.state.heading),
            math.cos(self.state.heading),
            t,
            self.progress - math.floor(self.progress),
            min(1.0, self.step_count / self.max_steps),
            1.0 if abs(nearest.signed_offset) > self.road_half_width else 0.0,
        ], dtype=np.float32)
        return np.clip(obs, -5.0, 5.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.state = CarState()
        p, tangent, _ = self._track_frame(0.0)
        self.state.x = float(p[0])
        self.state.z = float(p[1])
        self.state.heading = math.atan2(tangent[0], tangent[1])
        self.state.speed = 8.0
        self.step_count = 0
        self.progress = 0.0
        self.prev_progress = 0.0
        self.lap_count = 0
        self.last_track_t = 0.0
        self.visited_segments = set()
        self.last_info = {}
        return self._obs(), {}

    def step(self, action: np.ndarray):
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        drift = float(np.clip(action[3], 0.0, 1.0))

        self.state.throttle = throttle
        self.state.brake = brake
        steer_limit = 0.72 * (0.35 + 0.65 * max(0.0, 1.0 - abs(self.state.speed) / 55.0))
        steer_target = steer_cmd * steer_limit
        steer_delta = steer_target - self.state.steer
        self.state.steer += steer_delta * min(1.0, self.dt * 7.0)

        nearest = self._nearest(self.state.x, self.state.z, around_t=self.last_track_t)
        self.last_track_t = nearest.t
        heading_track = math.atan2(nearest.tangent[0], nearest.tangent[1])
        heading_err = self._wrap_angle(self.state.heading - heading_track)
        grip = self.base_grip - drift * 1.7

        a_long = self.accel_rate * throttle
        a_long -= self.brake_rate * brake * (1.0 if self.state.speed >= 0 else -1.0)
        a_long -= self.drag_linear * self.state.speed
        a_long -= self.drag_quad * self.state.speed * abs(self.state.speed)
        if abs(nearest.signed_offset) > self.road_half_width:
            a_long -= 10.0 * (1.0 if self.state.speed >= 0 else -1.0)

        desired_lat = np.clip(-heading_err * abs(self.state.speed) * self.corner_stiffness, -24.0, 24.0) * (0.65 + drift * 0.95)
        self.state.lateral_speed += (desired_lat - self.state.lateral_speed) * min(1.0, self.dt * (2.0 + drift))
        self.state.speed = float(np.clip(self.state.speed + a_long * self.dt, self.max_reverse, self.max_speed))
        self.state.lateral_speed *= math.exp(-grip * self.dt)

        yaw_lead = drift * np.clip(abs(self.state.lateral_speed) / 28.0, 0.0, 0.35) * np.sign(heading_err or 1.0)
        target_yaw = (self.state.speed / 3.5) * math.tan(self.state.steer + yaw_lead) if abs(self.state.speed) > 0.4 else 0.0
        self.state.yaw_rate += (target_yaw - self.state.yaw_rate) * min(1.0, self.dt * (4.8 + drift))
        self.state.yaw_rate *= math.exp(-(1.9 - drift * 0.4) * self.dt)
        self.state.heading = self._wrap_angle(self.state.heading + self.state.yaw_rate * self.dt)

        fwd = np.array([math.sin(self.state.heading), math.cos(self.state.heading)], dtype=np.float32)
        right_vec = np.array([fwd[1], -fwd[0]], dtype=np.float32)
        vel = fwd * self.state.speed + right_vec * self.state.lateral_speed
        self.state.x += float(vel[0] * self.dt)
        self.state.z += float(vel[1] * self.dt)

        nearest2 = self._nearest(self.state.x, self.state.z, around_t=self.last_track_t)
        self.last_track_t = nearest2.t
        self.prev_progress = self.progress
        segment_idx = int(nearest2.t * 8) % 8
        self.visited_segments.add(segment_idx)
        base_progress = self.lap_count + nearest2.t
        if base_progress + 0.5 < self.progress:
            self.lap_count += 1
            self.visited_segments.clear()
            base_progress = self.lap_count + nearest2.t
        self.progress = base_progress

        heading_track2 = math.atan2(nearest2.tangent[0], nearest2.tangent[1])
        heading_err2 = self._wrap_angle(self.state.heading - heading_track2)
        progress_delta = self.progress - self.prev_progress
        offroad = abs(nearest2.signed_offset) > self.road_half_width
        road_ratio = abs(nearest2.signed_offset) / self.road_half_width
        spin_penalty = max(0.0, abs(heading_err2) - 0.55)
        slip_angle = abs(math.atan2(self.state.lateral_speed, max(1.0, abs(self.state.speed))))
        control_effort = abs(steer_delta) + 0.3 * throttle + 0.45 * brake
        brake_on_straight = brake * max(0.0, 0.18 - abs(heading_err2)) * max(0.0, self.state.speed / self.max_speed)
        unstable_drift = drift * max(0.0, 0.12 - abs(heading_err2)) * (0.6 + max(0.0, self.state.speed) / self.max_speed)

        reward = (
            progress_delta * 120.0
            + max(0.0, self.state.speed) * 0.018
            - road_ratio * 0.18
            - offroad * 3.0
            - spin_penalty * 2.2
            - slip_angle * 0.45
            - abs(self.state.yaw_rate) * 0.025
            - control_effort * 0.03
            - brake_on_straight * 0.8
            - unstable_drift * 0.55
        )

        terminated = False
        truncated = False
        if self.lap_count >= 1:
            reward += 50.0
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
        if abs(heading_err2) > 2.6 and self.state.speed > 14.0:
            reward -= 10.0
            terminated = True
        if abs(nearest2.signed_offset) > self.road_half_width * 2.4:
            reward -= 12.0
            terminated = True

        self.step_count += 1
        info = {
            "track_progress": self.progress,
            "lap_count": self.lap_count,
            "track_t": nearest2.t,
            "signed_offset": nearest2.signed_offset,
            "distance_from_center": nearest2.distance,
            "heading_error": heading_err2,
            "slip_angle": slip_angle,
            "speed": self.state.speed,
            "lateral_speed": self.state.lateral_speed,
            "offroad": offroad,
            "segment_idx": segment_idx,
        }
        self.last_info = info
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        print({
            "step": self.step_count,
            "x": round(self.state.x, 2),
            "z": round(self.state.z, 2),
            "speed": round(self.state.speed, 2),
            "progress": round(self.progress, 3),
            **self.last_info,
        })

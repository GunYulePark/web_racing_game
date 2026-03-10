from __future__ import annotations

import unittest

import numpy as np

from racing_env import RacingEnv
from track_geometry import TrackModel
from track_data import build_tracks


class TrackGeometryTests(unittest.TestCase):
    def test_local_nearest_matches_global_on_track(self):
        track = TrackModel(build_tracks()["stadium"].points)
        point, _, _ = track.frame(0.37)
        local_hit = track.nearest(float(point[0]), float(point[1]), around_t=0.35)
        global_hit = track.nearest(float(point[0]), float(point[1]), around_t=None)
        self.assertAlmostEqual(local_hit.t, global_hit.t, places=2)
        self.assertLess(local_hit.distance, 2.0)


class RacingEnvTests(unittest.TestCase):
    def test_reset_and_step_shapes(self):
        env = RacingEnv(track="stadium")
        obs, info = env.reset(seed=123)
        self.assertEqual(obs.shape, (20,))
        self.assertEqual(info, {})
        next_obs, reward, terminated, truncated, step_info = env.step(np.array([0.0, 0.4, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(next_obs.shape, (20,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("track_progress", step_info)
        self.assertIn("slip_angle", step_info)

    def test_progress_stays_non_negative_under_heuristic(self):
        env = RacingEnv(track="stadium")
        obs, _ = env.reset(seed=7)
        last_progress = 0.0
        for _ in range(25):
            heading_err = float(obs[7])
            steer = np.clip(-heading_err * 1.5, -1.0, 1.0)
            action = np.array([steer, 0.65, 0.0, 0.0], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            self.assertGreaterEqual(info["track_progress"], last_progress - 0.05)
            last_progress = info["track_progress"]
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()

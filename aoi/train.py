"""Continuing PPO/DQN training that retains only the best verified schedule."""
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from .blocks import ConnectedBlockEnv
from .env import AoIEnv
from .policies import MaskedActorCriticPolicy, MaskedDQN
from .policy_eval import evaluate_policy
from .schedules import positive_integer


def make_model(env, *, algorithm="ppo", connected_block=False, gamma=0.999,
               seed=0, device="cpu", **algorithm_options):
    """Construct SB3; gamma and other numerical settings do not change the model."""
    if algorithm not in ("ppo", "dqn"):
        raise ValueError("Algorithm must be ppo or dqn.")
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Gamma must be in (0, 1].")
    if connected_block != isinstance(env.unwrapped, ConnectedBlockEnv):
        raise ValueError("Policy restriction and environment must agree.")
    cls = PPO if algorithm == "ppo" else (MaskedDQN if connected_block else DQN)
    policy = MaskedActorCriticPolicy if algorithm == "ppo" and connected_block else "MlpPolicy"
    return cls(policy, env, gamma=gamma, seed=seed, device=device, **algorithm_options)


class BestSchedule(BaseCallback):
    """Retain one verified result in memory, preferring lower exact average age."""

    def __init__(self, env_factory, *, validation_every=100_000, max_steps=50_000):
        super().__init__()
        if (isinstance(validation_every, bool) or not isinstance(validation_every, int)
                or validation_every < 0):
            raise ValueError("validation_every must be a nonnegative integer.")
        self.env_factory = env_factory
        self.every = validation_every
        self.max_steps = positive_integer(max_steps, "max_steps")
        self.result = None

    def _validate(self):
        env = self.env_factory()
        try:
            candidate = evaluate_policy(self.model, env, max_steps=self.max_steps)
        finally:
            env.close()
        if (self.result is None or self.result.schedule is None
                or (candidate.schedule is not None
                    and candidate.schedule.metrics.continuous_average_aoi
                    < self.result.schedule.metrics.continuous_average_aoi)):
            self.result = candidate

    def _on_step(self):
        if self.every and self.num_timesteps % self.every == 0:
            self._validate()
        return True

    def _on_training_end(self):
        # Validate after the final optimizer update, even if the step count is unchanged.
        self._validate()


def train(adjacency, output="result.json", *, algorithm="ppo", connected_block=False,
          timesteps=1_000_000, seed=0, gamma=0.999, validation_every=100_000,
          max_steps=50_000, device="cpu", algorithm_options=None):
    """Train with raw mean-age reward and write one selected schedule JSON file."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output file already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    positive_integer(timesteps, "timesteps")
    env_type = ConnectedBlockEnv if connected_block else AoIEnv
    env = env_type(adjacency)
    try:
        callback = BestSchedule(lambda: env_type(env.model.adjacency),
                                validation_every=validation_every, max_steps=max_steps)
        model = make_model(env, algorithm=algorithm, connected_block=connected_block,
                           gamma=gamma, seed=seed, device=device, **(algorithm_options or {}))
        model.learn(total_timesteps=timesteps, callback=callback)
        with output.open("x") as stream:
            stream.write(json.dumps(callback.result.to_dict(), indent=2, allow_nan=False) + "\n")
        return callback.result
    finally:
        env.close()

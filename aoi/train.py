"""Continuing PPO/DQN training and exact average/peak schedule selection."""
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from .env import AoIEnv
from .graphs import graph_to_dict, write_json
from .observations import PolicyObservation, default_divisor, parse_divisor
from .policies import MaskedActorCriticPolicy, MaskedDQN, MaskedDQNPolicy
from .policy_eval import evaluate_policy
from .schedules import ScheduleResult, positive_integer, schedule_entry


def parse_average_window(value):
    """Parse decimal text exactly; Python floats use their decimal spelling."""
    try:
        window = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("average_window must be a nonnegative finite number.") from exc
    if window < 0:
        raise ValueError("average_window must be a nonnegative finite number.")
    return window


class TrainingReward(gym.RewardWrapper):
    """Scale only the learner's reward; physical state and info remain raw."""

    def __init__(self, env, divisor=None):
        super().__init__(env)
        # N*diam(G) is a numerical conditioning heuristic. Positive constant
        # scaling preserves reward ordering, but can affect the learner's updates.
        self.divisor = (default_divisor(env.unwrapped.model)
                        if divisor is None else parse_divisor(divisor, "reward_divisor"))

    def reward(self, reward):
        return reward / self.divisor


def metric_point(schedule):
    return (schedule.metrics.continuous_average_aoi,
            schedule.metrics.continuous_peak_aoi)


@dataclass
class TrainingResult:
    """Nondominated verified schedules within a window of the best discovered average."""

    schedules: list[ScheduleResult] = field(default_factory=list)
    graph: dict | None = None
    method: str = "ppo"
    training: dict | None = None

    @property
    def status(self):
        return "cycle" if self.schedules else "evaluation_limit"

    def insert(self, candidate, window):
        # For a fixed window, best+window only decreases, so excluded points cannot
        # re-enter. Transitivity preserves dominance when a better point replaces
        # a dominator. Thus pruning preserves the frontier of discovered words.
        point = metric_point(candidate)
        # Weak dominance also deduplicates identical points (first wins).
        if any(a <= point[0] and p <= point[1]
               for a, p in map(metric_point, self.schedules)):
            return
        best = min(point[0], self.schedules[0].metrics.continuous_average_aoi
                   if self.schedules else point[0])
        if point[0] > best + window:
            return
        self.schedules[:] = [s for s in self.schedules
            if s.metrics.continuous_average_aoi <= best + window
            and not (point[0] <= s.metrics.continuous_average_aoi
                     and point[1] <= s.metrics.continuous_peak_aoi)]
        self.schedules.append(candidate)
        self.schedules.sort(key=metric_point)

    def to_dict(self):
        data = {"graph": self.graph,
                "schedules": [schedule_entry(s, self.method) for s in self.schedules]}
        if self.training is not None:
            data["training"] = dict(self.training)
        if not self.schedules:
            data["status"] = self.status
        return data


def make_model(env, *, algorithm="ppo", progress_mask=False, gamma=0.999,
               seed=0, device="cpu", reward_divisor=None, observation_divisor=None,
               **algorithm_options):
    """Construct PPO or DQN with independent scaling and optional progress masking.

    Pass a physical AoIEnv. The returned model records its effective settings so
    evaluate_policy applies the same observation convention to a fresh AoIEnv.
    """
    if algorithm not in ("ppo", "dqn"):
        raise ValueError("Algorithm must be ppo or dqn.")
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Gamma must be in (0, 1].")
    observations = PolicyObservation(env, observation_divisor, progress_mask=progress_mask)
    training_env = TrainingReward(observations, reward_divisor)
    cls = PPO if algorithm == "ppo" else (MaskedDQN if progress_mask else DQN)
    policy = ((MaskedActorCriticPolicy if algorithm == "ppo" else MaskedDQNPolicy)
              if progress_mask else "MlpPolicy")
    model = cls(policy, training_env, gamma=gamma, seed=seed, device=device,
                **algorithm_options)
    model.aoi_training = {"algorithm": algorithm, "reward_divisor": training_env.divisor,
                          "observation_divisor": observations.divisor,
                          "progress_mask": bool(progress_mask)}
    return model


class ScheduleFrontier(BaseCallback):
    """Validate separately and retain only useful average/peak tradeoffs."""

    def __init__(self, env_factory, *, validation_every=25_000, max_steps=50_000,
                 average_window=1):
        super().__init__()
        if (isinstance(validation_every, bool) or not isinstance(validation_every, int)
                or validation_every < 0):
            raise ValueError("validation_every must be a nonnegative integer.")
        self.env_factory = env_factory
        self.every = validation_every
        self.max_steps = positive_integer(max_steps, "max_steps")
        self.window = parse_average_window(average_window)
        self.result = TrainingResult()

    def _validate(self):
        # A fresh environment keeps validation separate from the training trajectory.
        env = self.env_factory()
        try:
            candidate = evaluate_policy(self.model, env, max_steps=self.max_steps)
        finally:
            env.close()
        if candidate.schedule is not None:
            self.result.insert(candidate.schedule, self.window)

    def _on_step(self):
        if self.every and self.num_timesteps % self.every == 0:
            self._validate()
        return True

    def _on_training_end(self):
        # Validate after the final optimizer update, even if the step count is unchanged.
        self._validate()


def train(adjacency, output="result.json", *, algorithm="ppo", progress_mask=False,
          timesteps=1_000_000, seed=0, gamma=0.999, validation_every=25_000,
          max_steps=50_000, device="cpu", reward_divisor=None, observation_divisor=None,
          average_window=1, algorithm_options=None):
    """Train PPO/DQN and return a TrainingResult, also written to new JSON output.

    ``adjacency`` follows AoIEnv's requirements; ``progress_mask`` optionally excludes
    relays that cannot improve a receiver. ``timesteps`` is the requested SB3
    budget (PPO may finish its rollout); ``algorithm_options`` passes extra SB3
    settings through.
    Both divisors default to N*diam(G), with independent positive overrides.
    Observation scaling also applies during deterministic policy validation.
    ``gamma`` discounts learner returns, not the exact physical metrics.

    ``validation_every=0`` requests final validation only; otherwise validation
    also runs at that step interval. ``max_steps`` bounds each policy evaluation.
    ``average_window`` retains nondominated discoveries within that exact rational
    distance of the best discovered average. This is not the global Pareto frontier.
    If no cycle is verified, the result has status ``evaluation_limit``.
    Existing output raises FileExistsError; a missing parent raises FileNotFoundError.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output file already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output parent directory does not exist: {output.parent}")
    positive_integer(timesteps, "timesteps")
    env = AoIEnv(adjacency)
    try:
        callback = ScheduleFrontier(lambda: AoIEnv(env.model.adjacency),
                                    validation_every=validation_every, max_steps=max_steps,
                                    average_window=average_window)
        callback.result.method = algorithm
        callback.result.graph = graph_to_dict(env.model.adjacency)
        model = make_model(env, algorithm=algorithm, progress_mask=progress_mask,
                           gamma=gamma, seed=seed, device=device, reward_divisor=reward_divisor,
                           observation_divisor=observation_divisor,
                           **(algorithm_options or {}))
        callback.result.training = dict(model.aoi_training)
        model.learn(total_timesteps=timesteps, callback=callback)
        write_json(callback.result.to_dict(), output)
        return callback.result
    finally:
        env.close()

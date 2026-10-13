"""Continuing PPO/DQN training with a small exact schedule frontier."""
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import gymnasium as gym
import networkx as nx
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from .blocks import ConnectedBlockEnv
from .env import AoIEnv
from .graphs import graph_to_dict
from .policies import MaskedActorCriticPolicy, MaskedDQN
from .policy_eval import evaluate_policy
from .schedules import ScheduleResult, positive_integer


def parse_reward_divisor(value):
    """Validate the finite positive numerical training scale."""
    try:
        divisor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reward_divisor must be a positive finite number.") from exc
    if isinstance(value, bool) or not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("reward_divisor must be a positive finite number.")
    return divisor


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
        self.divisor = (env.unwrapped.n * nx.diameter(
            nx.from_numpy_array(env.unwrapped.model.adjacency))
            if divisor is None else parse_reward_divisor(divisor))

    def reward(self, reward):
        return reward / self.divisor


def metric_point(schedule):
    return (schedule.metrics.continuous_average_aoi,
            schedule.metrics.continuous_peak_aoi)


@dataclass
class TrainingResult:
    """Only nondominated verified words in the current exact average window."""

    schedules: list[ScheduleResult] = field(default_factory=list)
    graph: dict | None = None

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
        graph = {"graph": self.graph} if self.graph is not None else {}
        if not self.schedules:
            return {**graph, "status": self.status}
        words = [{"cycle_actions": [list(a) for a in s.actions], **s.metrics.to_dict()}
                 for s in self.schedules]
        return {**graph, "status": self.status, "best_average": words[0], "alternatives": words[1:]}


def make_model(env, *, algorithm="ppo", connected_block=False, gamma=0.999,
               seed=0, device="cpu", reward_divisor=None, **algorithm_options):
    """Construct SB3 with training-only reward scaling and unchanged masks."""
    if algorithm not in ("ppo", "dqn"):
        raise ValueError("Algorithm must be ppo or dqn.")
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("Gamma must be in (0, 1].")
    if connected_block != isinstance(env.unwrapped, ConnectedBlockEnv):
        raise ValueError("Policy restriction and environment must agree.")
    cls = PPO if algorithm == "ppo" else (MaskedDQN if connected_block else DQN)
    policy = MaskedActorCriticPolicy if algorithm == "ppo" and connected_block else "MlpPolicy"
    return cls(policy, TrainingReward(env, reward_divisor), gamma=gamma,
               seed=seed, device=device, **algorithm_options)


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
        # A fresh, unscaled environment isolates resets and block memory from the
        # live rollout. Deterministic prediction avoids exploration RNG draws;
        # exact evaluation measures physical ages, independent of reward scaling.
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


def train(adjacency, output="result.json", *, algorithm="ppo", connected_block=False,
          timesteps=1_000_000, seed=0, gamma=0.999, validation_every=25_000,
          max_steps=50_000, device="cpu", reward_divisor=None, average_window=1,
          algorithm_options=None):
    """Train PPO/DQN and return a TrainingResult, also written to new JSON output.

    ``adjacency`` follows AoIEnv's requirements; ``connected_block`` selects the
    heuristic restriction. ``timesteps`` is the requested SB3 budget (PPO may
    finish its rollout); ``algorithm_options`` passes extra SB3 settings through.
    ``reward_divisor=None`` uses N*diam(G); a positive override affects training
    only. ``gamma`` discounts learner returns, not the exact physical metrics.

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
    env_type = ConnectedBlockEnv if connected_block else AoIEnv
    env = env_type(adjacency)
    try:
        callback = ScheduleFrontier(lambda: env_type(env.model.adjacency),
                                    validation_every=validation_every, max_steps=max_steps,
                                    average_window=average_window)
        callback.result.graph = graph_to_dict(env.model.adjacency)
        model = make_model(env, algorithm=algorithm, connected_block=connected_block,
                           gamma=gamma, seed=seed, device=device, reward_divisor=reward_divisor,
                           **(algorithm_options or {}))
        model.learn(total_timesteps=timesteps, callback=callback)
        with output.open("x") as stream:
            stream.write(json.dumps(callback.result.to_dict(), indent=2, allow_nan=False) + "\n")
        return callback.result
    finally:
        env.close()

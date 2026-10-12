"""Reliable, one-transmitter-per-slot Age of Information dynamics."""
from __future__ import annotations

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces


def validate_adjacency(adjacency) -> np.ndarray:
    """Return a binary, connected, undirected adjacency matrix without relabeling."""
    a = np.asarray(adjacency)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("Adjacency must be square with at least two nodes.")
    if a.dtype.kind not in "biuf" or not np.isin(a, (0, 1)).all():
        raise ValueError("Adjacency must contain only finite zero/one entries.")
    a = a.astype(bool, copy=True)
    if np.diag(a).any() or not np.array_equal(a, a.T):
        raise ValueError("Adjacency must be symmetric with zero diagonal.")
    if not nx.is_connected(nx.from_numpy_array(a)):
        raise ValueError("Graph must be connected.")
    return a


class AoIModel:
    """Pure integer dynamics; state entries are (monitor, process), row major."""

    def __init__(self, adjacency):
        self.adjacency = validate_adjacency(adjacency)
        self.n = len(self.adjacency)
        self.pairs = tuple((m, j) for m in range(self.n)
                           for j in range(self.n) if m != j)
        self.lookup = np.full((self.n, self.n), -1, dtype=np.int64)
        for k, (m, j) in enumerate(self.pairs):
            self.lookup[m, j] = k
        self.neighbors = tuple(np.flatnonzero(row) for row in self.adjacency)
        distances = dict(nx.all_pairs_shortest_path_length(
            nx.from_numpy_array(self.adjacency)))
        self.initial_state = np.array([distances[m][j] for m, j in self.pairs],
                                      dtype=np.int64)

    def action_index(self, pair) -> int:
        if len(pair) != 2 or any(isinstance(x, (bool, np.bool_))
                                or not isinstance(x, (int, np.integer)) for x in pair):
            raise ValueError("An action must contain two integer node indices.")
        i, j = map(int, pair)
        if not (0 <= i < self.n and 0 <= j < self.n):
            raise ValueError("Action node index is out of range.")
        return i * self.n + j

    def action_pair(self, action: int) -> tuple[int, int]:
        if (isinstance(action, (bool, np.bool_))
                or not isinstance(action, (int, np.integer))
                or not 0 <= action < self.n * self.n):
            raise ValueError("Discrete action is out of range.")
        return divmod(int(action), self.n)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Broadcast the transmitter's pre-slot copy, then return post-slot ages."""
        i, j = self.action_pair(action)
        x = np.asarray(state)
        if (x.shape != self.initial_state.shape or x.dtype.kind not in "iu"
                or np.any(x < 0)):
            raise ValueError("State must be a nonnegative integer age vector.")
        if int(x.max()) >= np.iinfo(np.int64).max:
            raise OverflowError("Age exceeds the supported integer range.")
        y = x.astype(np.int64, copy=True) + 1
        relay_age = 1 if i == j else int(x[self.lookup[i, j]]) + 1
        monitors = self.neighbors[i]
        indices = self.lookup[monitors[monitors != j], j]
        y[indices] = np.minimum(y[indices], relay_age)
        return y


class AoIEnv(gym.Env):
    """Continuing all-process AoI environment with all N² actions.

    Reset uses shortest-path ages. Observations are unnormalized float32 ages;
    the mathematical state remains integer valued. Reward is negative post-slot
    average age. There is no natural termination or truncation.
    """

    metadata = {"render_modes": []}

    def __init__(self, adjacency):
        super().__init__()
        self.model = AoIModel(adjacency)
        self.n = self.model.n
        self.action_space = spaces.Discrete(self.n * self.n)
        self.observation_space = spaces.Box(
            0, np.inf, shape=self.model.initial_state.shape, dtype=np.float32)
        self.state = self.model.initial_state.copy()

    def action_index(self, pair) -> int:
        return self.model.action_index(pair)

    def action_pair(self, action: int) -> tuple[int, int]:
        return self.model.action_pair(action)

    def observation(self) -> np.ndarray:
        return self.state.astype(np.float32, copy=True)

    def state_key(self) -> tuple[int, ...]:
        return tuple(map(int, self.state))

    def info(self) -> dict:
        average = float(np.mean(self.state))
        return {"average_age": average, "peak_age": int(self.state.max()),
                "raw_reward": -average}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.model.initial_state.copy()
        return self.observation(), self.info()

    def step(self, action):
        self.state = self.model.transition(self.state, action)
        info = self.info()
        return self.observation(), info["raw_reward"], False, False, info

"""Source-started connected-block action restriction with explicit memory."""
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .env import AoIEnv, AoIModel
from .schedules import validate_word


@dataclass(frozen=True)
class BlockMemory:
    active: int = -1
    holders: int = 0


class BlockRule:
    """Track holders of the update generated at the start of the current block."""

    def __init__(self, model: AoIModel):
        self.n = model.n
        self.neighbors = tuple(sum(1 << int(v) for v in row) for row in model.neighbors)

    def legal(self, memory: BlockMemory, pair) -> bool:
        i, j = pair
        return i == j or (j == memory.active and bool(memory.holders & (1 << i)))

    def advance(self, memory: BlockMemory, pair) -> BlockMemory:
        if not self.legal(memory, pair):
            raise ValueError("Action violates connected-block legality.")
        i, j = pair
        # Every source action starts a new update, even for the same process.
        # Holders are a bit set of nodes reached by this block's update; only
        # these nodes may relay it, and previous branches remain available.
        holders = (1 << j) if i == j else memory.holders
        return BlockMemory(j, holders | self.neighbors[i])

    def mask(self, memory: BlockMemory) -> np.ndarray:
        return np.array([self.legal(memory, (i, j)) for i in range(self.n)
                         for j in range(self.n)], dtype=bool)

    def encode(self, memory: BlockMemory) -> np.ndarray:
        active = np.zeros(self.n, dtype=np.float32)
        if memory.active >= 0:
            active[memory.active] = 1
        holders = np.array([(memory.holders >> i) & 1 for i in range(self.n)],
                           dtype=np.float32)
        return np.concatenate((active, holders))


class ConnectedBlockEnv(AoIEnv):
    """Apply a source-started search heuristic to the AoIEnv physical model.

    ``adjacency`` has the same requirements as AoIEnv. Observations append an
    N-entry active-source one-hot vector and N holder bits to the physical ages.
    Reset has no active block, so only sources may transmit initially. Later,
    any source may restart a block; relays must hold the active block's update.
    Illegal actions raise ValueError. This restriction is not proved lossless.

    Block memory records update membership, which ages alone do not encode;
    it is needed both for action masks and for controlled-state recurrence.
    """

    def __init__(self, adjacency):
        super().__init__(adjacency)
        self.rule = BlockRule(self.model)
        self.memory = BlockMemory()
        self.observation_space = spaces.Box(
            0, np.inf, shape=(len(self.state) + 2 * self.n,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.memory = BlockMemory()
        return super().reset(seed=seed, options=options)

    def observation(self):
        return np.concatenate((super().observation(), self.rule.encode(self.memory)))

    def action_masks(self):
        return self.rule.mask(self.memory)

    def state_key(self):
        return super().state_key() + (self.memory.active, self.memory.holders)

    def step(self, action):
        pair = self.action_pair(action)
        memory = self.rule.advance(self.memory, pair)
        state = self.model.transition(self.state, action)
        self.memory, self.state = memory, state
        info = self.info()
        return self.observation(), info['raw_reward'], False, False, info


def is_cyclic_block_schedule(adjacency, actions) -> bool:
    """Check blocks across the word boundary, independently of its stored phase."""
    model = AoIModel(adjacency)
    word = validate_word(model, actions)
    start = next((k for k, (i, j) in enumerate(word) if i == j), None)
    if start is None:
        return False
    word = word[start:] + word[:start]
    rule, memory = BlockRule(model), BlockMemory()
    for pair in word:
        if not rule.legal(memory, pair):
            return False
        memory = rule.advance(memory, pair)
    return True

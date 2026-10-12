import networkx as nx
import numpy as np
import pytest
from aoi.blocks import BlockMemory, ConnectedBlockEnv, is_cyclic_block_schedule
from aoi.env import AoIEnv


def test_legality_memory_and_unchanged_physics():
    a = nx.to_numpy_array(nx.path_graph(4))
    env, physical = ConnectedBlockEnv(a), AoIEnv(a)
    assert np.flatnonzero(env.action_masks()).tolist() == [0, 5, 10, 15]
    actions = [(0, 0), (1, 0), (2, 0), (1, 0)]
    for pair in actions:
        action = env.action_index(pair)
        env.step(action); physical.step(action)
        np.testing.assert_array_equal(env.state, physical.state)
    assert env.memory == BlockMemory(0, 15)
    env.step(0)  # A fresh update discards distant holders, even for the same source.
    assert env.memory == BlockMemory(0, 3)
    state, memory = env.state.copy(), env.memory
    with pytest.raises(ValueError):
        env.step(env.action_index((2, 0)))
    np.testing.assert_array_equal(env.state, state)
    assert env.memory == memory
    env.step(env.action_index((3, 3)))
    assert not env.action_masks()[env.action_index((1, 0))]
    env.reset()
    assert env.memory == BlockMemory()


def test_cyclic_wrap_and_branches():
    graph = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
    a = nx.to_numpy_array(graph, nodelist=range(5))
    word = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 0))
    for phase in range(len(word)):
        assert is_cyclic_block_schedule(a, word[phase:] + word[:phase])
    assert not is_cyclic_block_schedule(a, [(1, 0)])
    assert not is_cyclic_block_schedule(a, [(0, 0), (0, 0), (3, 0)])


def test_integer_key_contains_memory_even_when_ages_match():
    env = ConnectedBlockEnv([[0, 1], [1, 0]])
    first = env.state_key()
    env.memory = BlockMemory(0, 3)
    assert env.state_key() != first
    obs = env.observation()
    assert obs[-4:].tolist() == [1, 0, 1, 1]

import networkx as nx
import numpy as np
import pytest

from aoi.env import AoIEnv, AoIModel
from aoi.observations import PolicyObservation, progress_mask


@pytest.mark.parametrize('graph', [nx.path_graph(3), nx.path_graph(7), nx.path_graph(8),
                                  nx.cycle_graph(5), nx.star_graph(5), nx.complete_graph(4)])
def test_mask_matches_all_direct_transitions_on_reachable_states(graph):
    model = AoIModel(nx.to_numpy_array(graph))
    rng = np.random.default_rng(192)
    state = model.initial_state.copy()
    for _ in range(250):
        before = state.copy()
        mask = progress_mask(model, state)
        assert mask.dtype == bool and mask.shape == (model.n**2,)
        assert mask.reshape(model.n, model.n).diagonal().all()
        assert mask.any()
        for action in range(model.n**2):
            i, j = model.action_pair(action)
            after = model.transition(state, action)
            assert mask[action] == (i == j or (after < state + 1).any())
        np.testing.assert_array_equal(state, before)  # Pure calculation.
        state = model.transition(state, int(rng.integers(model.n**2)))


def test_equal_stale_and_multiple_useful_receivers():
    model = AoIModel(nx.to_numpy_array(nx.path_graph(4)))
    state = np.full(12, 5, dtype=np.int64)
    assert np.flatnonzero(progress_mask(model, state)).tolist() == [0, 5, 10, 15]
    # Source receiver 0 is excluded; relay 1 can improve receiver 2 only.
    for value, expected in [(4, True), (5, False), (6, False)]:
        state[model.lookup[1, 0]] = value
        assert progress_mask(model, state)[4] == expected
    # Two neighbors of transmitter 2 can both receive fresher source-0 copies.
    state[model.lookup[1, 0]] = 5
    state[model.lookup[2, 0]] = 4
    assert progress_mask(model, state)[8]
    assert (model.transition(state, 8) < state + 1).sum() == 2


def test_interruption_resumption_consecutive_generations_and_no_fairness():
    env = AoIEnv(nx.to_numpy_array(nx.path_graph(7)))
    sequence = [(0, 0), (6, 6), (1, 0), (5, 6), (2, 0), (4, 6)]
    sequence += [(3, 3)] * 30  # No deadline for any other source.
    sequence += [(3, 0), (3, 6)]  # Revisit both partially disseminated updates.
    for pair in sequence:
        action = env.action_index(pair)
        assert progress_mask(env.model, env.state)[action]
        env.step(action)


@pytest.mark.parametrize('divisor', [1, 42, 1e40])
def test_masks_do_not_depend_on_rounded_or_scaled_ages(divisor):
    env = AoIEnv(nx.to_numpy_array(nx.path_graph(7)))
    env.state[:] = 2**28
    env.state[env.model.lookup[1, 0]] -= 1
    wrapped = PolicyObservation(env, divisor, progress_mask=True)
    obs = wrapped.observation(env.observation())
    assert obs['ages'][env.model.lookup[1, 0]] == obs['ages'][env.model.lookup[2, 0]]
    assert obs['mask'][7]
    np.testing.assert_array_equal(obs['mask'], progress_mask(env.model, env.state))
    assert len(obs['ages']) == 42
    assert len(env.state_key()) == 42


@pytest.mark.parametrize('state', [np.ones(6), np.ones(5, dtype=int), -np.ones(6, dtype=int)])
def test_mask_rejects_nonphysical_states(state):
    with pytest.raises(ValueError, match='integer age vector'):
        progress_mask(AoIModel(nx.to_numpy_array(nx.path_graph(3))), state)

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from aoi.env import AoIEnv, AoIModel, validate_adjacency


def path(n):
    return (np.abs(np.arange(n)[:, None] - np.arange(n)) == 1).astype(int)


@pytest.mark.parametrize('a', [np.zeros((1, 1)), np.zeros((2, 3)),
    [[0, 1], [0, 0]], [[1, 1], [1, 0]], np.zeros((3, 3)),
    [[0, np.nan], [np.nan, 0]], [[0, 2], [2, 0]]])
def test_invalid_graph(a):
    with pytest.raises(ValueError):
        validate_adjacency(a)


def test_initialization_actions_and_gym():
    env = AoIEnv(path(4))
    check_env(env, skip_render_check=True)
    assert env.action_space.n == 16
    assert env.model.pairs == tuple((m, j) for m in range(4) for j in range(4) if m != j)
    assert env.state.tolist() == [abs(m-j) for m, j in env.model.pairs]
    for a in range(16):
        assert env.action_index(env.action_pair(a)) == a
    for a in (-1, 16, 1.0, True):
        with pytest.raises(ValueError):
            env.action_pair(a)


def test_source_relay_and_simultaneous_updates():
    model = AoIModel(path(4))
    x = np.full(12, 20, dtype=np.int64)
    y = model.transition(x, model.action_index((0, 0)))
    assert y[model.lookup[1, 0]] == 1
    assert y[model.lookup[2, 0]] == 21  # No forwarding within the same slot.
    z = model.transition(y, model.action_index((1, 0)))
    assert z[model.lookup[2, 0]] == 2
    assert z[model.lookup[3, 0]] == 22
    np.testing.assert_array_equal(x, np.full(12, 20))
    stale = model.transition(z, model.action_index((3, 0)))
    assert stale[model.lookup[2, 0]] == 3


def test_reward_is_post_slot_and_process_continues():
    env = AoIEnv(path(3))
    for _ in range(12):
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated and not truncated
        assert reward == -float(np.mean(env.state)) == info['raw_reward']
        assert obs.dtype == np.float32
    assert env.state.max() > 12


def test_integer_overflow_is_not_recurrence():
    model = AoIModel(path(2))
    with pytest.raises(OverflowError):
        model.transition(np.full(2, np.iinfo(np.int64).max), 0)

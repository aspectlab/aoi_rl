import json

import networkx as nx
import numpy as np
import pytest
import torch

from aoi.env import AoIEnv
from aoi.observations import PolicyObservation, default_divisor
from aoi.train import TrainingReward, make_model, train


@pytest.mark.parametrize('graph', [nx.path_graph(7), nx.path_graph(8), nx.cycle_graph(6),
                                  nx.star_graph(4), nx.complete_graph(5)])
def test_both_default_divisors_are_n_times_diameter(graph):
    env = AoIEnv(nx.to_numpy_array(graph))
    expected = len(graph) * nx.diameter(graph)
    assert default_divisor(env.model) == expected
    assert TrainingReward(env).divisor == PolicyObservation(env).divisor == expected


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('reward,observation', [(None, None), (1, None), (None, 1),
                                               (1, 1), (3.5, 17), (None, 1e40)])
def test_independent_scales_reach_networks_and_preserve_physics(algorithm, masked, reward, observation):
    adjacency = nx.to_numpy_array(nx.path_graph(4))
    env, physical = AoIEnv(adjacency), AoIEnv(adjacency)
    kwargs = dict(n_steps=8, batch_size=4) if algorithm == 'ppo' else dict(buffer_size=16)
    model = make_model(env, algorithm=algorithm, progress_mask=masked,
                       reward_divisor=reward, observation_divisor=observation, **kwargs)
    obs_scale = 12 if observation is None else observation
    reward_scale = 12 if reward is None else reward
    received = []
    # Verify the actual first neural layer receives only N(N-1) scaled ages.
    layer = (model.policy.mlp_extractor.policy_net[0] if algorithm == 'ppo'
             else model.q_net.q_net[0])
    handle = layer.register_forward_pre_hook(lambda module, args: received.append(args[0].detach().clone()))
    vec = model.env
    vec.reset()
    for action in [0, 4, 15, 11, 5, 1, 8, 2]:
        _, raw, _, _, _ = physical.step(action)
        obs, rewards, dones, infos = vec.step(np.array([action]))
        ages = obs['ages'] if masked else obs
        expected = (physical.state.astype(np.float64) / obs_scale).astype(np.float32)
        np.testing.assert_array_equal(ages[0], expected)
        np.testing.assert_array_equal(env.state, physical.state)
        assert env.state_key() == physical.state_key()
        assert rewards[0] == np.float32(raw / reward_scale)
        assert infos[0]['raw_reward'] == raw
        assert not dones[0]
        model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(received[-1][0], expected)
        assert received[-1].shape == (1, 12)
    handle.remove()
    assert model.aoi_training == dict(algorithm=algorithm, reward_divisor=reward_scale,
                                      observation_divisor=obs_scale, progress_mask=masked)


@pytest.mark.parametrize('name', ['reward_divisor', 'observation_divisor'])
@pytest.mark.parametrize('value', [0, -1, 'nan', 'inf', '-inf', 'oops', '', '1e999', True, np.bool_(True)])
def test_invalid_divisors_fail_before_model_construction(name, value):
    with pytest.raises(ValueError, match=name + ' must be a positive finite number'):
        make_model(AoIEnv([[0, 1], [1, 0]]), **{name: value})


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('reward,observation', [(None, None), (1, None), (None, 1), (1, 56)])
def test_effective_training_json_scales(tmp_path, algorithm, masked, reward, observation):
    options = (dict(n_steps=8, batch_size=4, n_epochs=1) if algorithm == 'ppo'
               else dict(buffer_size=16, learning_starts=0, batch_size=4, train_freq=1))
    output = tmp_path / 'result.json'
    result = train(nx.to_numpy_array(nx.path_graph(3)), output, algorithm=algorithm,
                   progress_mask=masked, reward_divisor=reward, observation_divisor=observation,
                   timesteps=8, max_steps=10, validation_every=4, algorithm_options=options)
    assert result.training == dict(algorithm=algorithm, reward_divisor=6 if reward is None else reward,
                                   observation_divisor=6 if observation is None else observation,
                                   progress_mask=masked)
    assert json.loads(output.read_text())['training'] == result.training

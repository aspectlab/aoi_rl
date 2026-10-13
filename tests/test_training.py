import importlib
import json
import pickle
import random

import networkx as nx
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback

from aoi.blocks import ConnectedBlockEnv
from aoi.env import AoIEnv
from aoi.policies import MaskedDQN, observation_mask
from aoi.train import (ScheduleFrontier, TrainingReward, TrainingResult,
                       parse_average_window, make_model, train)
from fractions import Fraction
from aoi.schedules import CycleMetrics, ScheduleResult
from aoi.schedules import evaluate_schedule


torch.set_num_threads(1)


def options(algorithm):
    return (dict(n_steps=8, batch_size=4, n_epochs=1) if algorithm == 'ppo' else
            dict(buffer_size=64, learning_starts=0, batch_size=4, train_freq=1,
                 target_update_interval=8))


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('restricted', [False, True])
@pytest.mark.parametrize('divisor', [None, 1, 26])
@pytest.mark.parametrize('graph', [nx.path_graph(4), nx.cycle_graph(4)])
def test_scaled_reward_and_separate_validation(algorithm, restricted, graph, divisor):
    cls = ConnectedBlockEnv if restricted else AoIEnv

    class CountResets(cls):
        resets = 0
        def reset(self, **kwargs):
            self.resets += 1
            return super().reset(**kwargs)

    a = nx.to_numpy_array(graph)
    base, reference = CountResets(a), cls(a)
    model = make_model(base, algorithm=algorithm, connected_block=restricted,
                       seed=6, reward_divisor=divisor, **options(algorithm))
    scale = len(a) * nx.diameter(graph) if divisor is None else divisor
    raw_rewards = []

    class Audit(BaseCallback):
        def _on_step(self):
            action = int(self.locals['actions'][0])
            obs, raw, terminated, truncated, _ = reference.step(action)
            assert raw == -float(np.mean(reference.state))
            assert not terminated and not truncated
            assert self.locals['rewards'][0] == np.float32(raw / scale)
            np.testing.assert_array_equal(self.locals['new_obs'][0], obs)
            assert base.state_key() == reference.state_key()
            assert not self.locals['dones'][0]
            assert 'terminal_observation' not in self.locals['infos'][0]
            assert base.resets == 1
            raw_rewards.append(raw / scale)
            return True

        def _on_rollout_end(self):
            if algorithm == 'ppo':
                np.testing.assert_array_equal(model.rollout_buffer.rewards[:, 0],
                                              np.asarray(raw_rewards[-8:], dtype=np.float32))
                if restricted:
                    obs = torch.tensor(model.rollout_buffer.observations[:, 0])
                    actions = torch.tensor(model.rollout_buffer.actions[:, 0, 0], dtype=torch.long)
                    _, logp, entropy = model.policy.evaluate_actions(obs, actions)
                    np.testing.assert_allclose(logp.detach(), model.rollout_buffer.log_probs[:, 0], atol=1e-6)
                    assert torch.isfinite(entropy).all()

    validation = ScheduleFrontier(lambda: cls(a), validation_every=4, max_steps=20)
    model.learn(24, callback=[Audit(), validation])
    assert base.resets == 1
    if algorithm == 'dqn':
        np.testing.assert_array_equal(model.replay_buffer.rewards[:24, 0],
                                      np.asarray(raw_rewards, dtype=np.float32))
        assert not model.replay_buffer.dones[:24].any()
        assert not model.replay_buffer.timeouts[:24].any()
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    model.env.close()


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('restricted', [False, True])
def test_train_writes_only_json_and_never_computes_mcds(tmp_path, monkeypatch, algorithm, restricted):
    def forbidden(*args, **kwargs):
        raise AssertionError('Training must not compute MCDS')
    monkeypatch.setattr(importlib.import_module('aoi.algorithm1'), 'algorithm1_schedule', forbidden)
    output = tmp_path/'result.json'
    result = train(nx.to_numpy_array(nx.path_graph(3)), output, algorithm=algorithm,
                   connected_block=restricted, timesteps=32, validation_every=16,
                   max_steps=50, algorithm_options=options(algorithm))
    assert list(tmp_path.iterdir()) == [output]
    assert json.loads(output.read_text()) == result.to_dict()
    assert result.status in ('cycle', 'evaluation_limit')
    for schedule in result.schedules:
        replay = evaluate_schedule(nx.to_numpy_array(nx.path_graph(3)), schedule.actions)
        assert replay.metrics == schedule.metrics
    assert 'transient' not in output.read_text()
    with pytest.raises(FileExistsError):
        train([[0, 1], [1, 0]], output, timesteps=1, max_steps=10)


def test_intermediate_best_survives_worse_final_policy():
    class Policy:
        mode = 'best'
        def predict(self, obs, deterministic=True):
            if self.mode == 'worse':
                return (0 if obs[1] >= 3 else 3), None
            if self.mode == 'tie':
                return (0 if obs[1] > obs[0] else 3), None
            return (0 if obs[1] >= obs[0] else 3), None

    model = Policy()
    callback = ScheduleFrontier(lambda: AoIEnv([[0, 1], [1, 0]]), validation_every=1, max_steps=20)
    callback.model, callback.num_timesteps = model, 1
    callback._on_step()
    best = callback.result.schedules[0]
    model.mode = 'tie'
    callback._on_step()
    assert callback.result.schedules[0] is best
    model.mode = 'worse'
    callback._on_training_end()
    assert callback.result.schedules[0] is best
    # Final validation can also discover an improvement after the last update.
    callback.result = TrainingResult()
    callback._on_step()
    worse = callback.result.schedules[0]
    model.mode = 'best'
    callback._on_training_end()
    assert callback.result.schedules[0].metrics.continuous_average_aoi < worse.metrics.continuous_average_aoi
def test_dqn_masks_targets_and_all_exploration_paths():
    env = ConnectedBlockEnv(nx.to_numpy_array(nx.path_graph(3)))
    model = MaskedDQN('MlpPolicy', env, seed=2, buffer_size=16, learning_starts=10)
    obs, _ = env.reset()
    for net in (model.q_net, model.q_net_target):
        for parameter in net.parameters():
            parameter.data.zero_()
        net.q_net[-1].bias.data[1] = 1e6
        net.q_net[-1].bias.data[0] = 2
    assert model.q_net_target(torch.tensor(obs[None])).max().item() == 2
    assert int(model.predict(obs, deterministic=True)[0]) == 0
    model._setup_learn(1)
    model.replay_buffer.add(obs[None], obs[None], np.array([0]), np.array([1.]), np.array([0.]), [{}])
    model.train(1, 1)
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor([2.]), torch.tensor([1 + 2*model.gamma])).item()
    assert model.logger.name_to_value['train/loss'] == pytest.approx(expected)
    model.exploration_rate = 1
    model._last_obs = obs[None]
    for _ in range(50):
        assert env.action_masks()[int(model.predict(obs)[0])]
        assert env.action_masks()[int(model._sample_action(10)[0][0])]

def test_ppo_masks_inference_sampling_likelihood_and_gradient():
    env = ConnectedBlockEnv(nx.to_numpy_array(nx.path_graph(3)))
    model = make_model(env, connected_block=True, **options('ppo'))
    obs = torch.tensor(env.reset()[0][None])
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[1] = 1e6
    distribution = model.policy.get_distribution(obs)
    mask = observation_mask(obs, 3)
    assert (distribution.distribution.probs[~mask] == 0).all()
    for _ in range(20):
        action, _, _ = model.policy(obs)
        assert env.action_masks()[int(action[0])]
    assert env.action_masks()[int(model.predict(obs.numpy(), deterministic=True)[0][0])]
    _, logp, entropy = model.policy.evaluate_actions(obs, torch.tensor([0]))
    model.policy.zero_grad()
    (-(logp + entropy).sum()).backward()
    assert model.policy.action_net.bias.grad[1] == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.policy.parameters())


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('restricted', [False, True])
def test_guaranteed_training_json_schedule_round_trip(tmp_path, monkeypatch, capsys, algorithm, restricted):
    adjacency = [[0, 1], [1, 0]]
    models = []

    def fixed_model(env, **kwargs):
        model = make_model(env, **kwargs)
        # Q(0, 0) = age at monitor 1; Q(1, 1) = age at monitor 0.
        # Zero learning rate preserves this known cycle through real SB3 updates.
        with torch.no_grad():
            layers = ([model.policy.action_net] if algorithm == 'ppo' else
                      [model.q_net.q_net[0], model.q_net_target.q_net[0]])
            for layer in layers:
                layer.weight.zero_()
                layer.bias.fill_(-10)
                layer.weight[0, 1] = layer.weight[3, 0] = 1
                layer.bias[0] = layer.bias[3] = 0
        models.append(model)
        return model

    monkeypatch.setattr(importlib.import_module('aoi.train'), 'make_model', fixed_model)
    output = tmp_path/'result.json'
    exploration = ({} if algorithm == 'ppo' else
                   {'exploration_initial_eps': 0, 'exploration_final_eps': 0})
    result = train(adjacency, output, algorithm=algorithm, connected_block=restricted, timesteps=8,
                   validation_every=4, max_steps=10, algorithm_options={
                       **options(algorithm), 'learning_rate': 0, **exploration,
                       'policy_kwargs': {'net_arch': []}})
    from aoi.__main__ import read_schedule
    assert read_schedule(output) == json.loads(output.read_text())['best_average']['cycle_actions']
    assert models[0]._n_updates == (1 if algorithm == 'ppo' else 8)
    data = json.loads(output.read_text())
    assert data == result.to_dict() == {
        'graph': {'num_nodes': 2, 'edges': [[0, 1]]},
        'status': 'cycle', 'best_average': {
            'cycle_actions': [[1, 1], [0, 0]], 'period': 2,
            'continuous_average_aoi': '2', 'continuous_peak_aoi': 3},
        'alternatives': []}
    replay = evaluate_schedule(adjacency, data['best_average']['cycle_actions'])
    assert replay.metrics == result.schedules[0].metrics
    assert replay.to_dict() == {'status': 'cycle', **data['best_average']}
    from aoi.__main__ import main
    assert main(['evaluate', str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == {'graph': data['graph'], **replay.to_dict()}


def test_deterministic_validation_preserves_training_rng_state():
    adjacency = [[0, 1], [1, 0]]
    for algorithm in ('ppo', 'dqn'):
        for restricted in (False, True):
            cls = ConnectedBlockEnv if restricted else AoIEnv
            env = cls(adjacency)
            model = make_model(env, algorithm=algorithm, connected_block=restricted,
                               seed=719, **options(algorithm))
            try:
                model.learn(8)
                validation = ScheduleFrontier(lambda: cls(adjacency), max_steps=10)
                validation.init_callback(model)
                python_state = random.getstate()
                numpy_state = pickle.dumps(np.random.get_state())
                torch_state = torch.get_rng_state().clone()
                env_rng = pickle.dumps(env.np_random.bit_generator.state)
                action_rng = pickle.dumps(env.action_space.np_random.bit_generator.state)
                validation._validate()
                assert random.getstate() == python_state
                assert pickle.dumps(np.random.get_state()) == numpy_state
                assert torch.equal(torch.get_rng_state(), torch_state)
                assert pickle.dumps(env.np_random.bit_generator.state) == env_rng
                assert pickle.dumps(env.action_space.np_random.bit_generator.state) == action_rng
            finally:
                model.env.close()


def synthetic(average, peak, action=0):
    return ScheduleResult(((action, action),), CycleMetrics(1, Fraction(average), peak))


def test_exact_frontier_pruning_order_and_deduplication():
    result = TrainingResult()
    first = synthetic('15.55', 33)
    for candidate in [first, synthetic('15.55', 33, 1), synthetic('16.5', 25),
                      synthetic('16.1', 29), synthetic('16', 34),
                      synthetic('15.9', 31), synthetic('15.3', 36), synthetic('15.3', 35)]:
        result.insert(candidate, Fraction(1))
    assert [(s.metrics.continuous_average_aoi, s.metrics.continuous_peak_aoi)
            for s in result.schedules] == [(Fraction(a), p) for a, p in
            [('15.3', 35), ('15.55', 33), ('15.9', 31), ('16.1', 29)]]
    assert result.schedules[1] is first
    result.insert(synthetic('15.1', 40), Fraction(1))
    assert result.schedules[-1].metrics.continuous_average_aoi == Fraction('16.1')
    result.insert(synthetic('15.09999999999999999999', 40), Fraction(1))
    assert result.schedules[-1].metrics.continuous_average_aoi == Fraction('15.9')
    result.insert(synthetic('14', 30), Fraction(1))
    assert result.schedules == [synthetic('14', 30)]
    assert len(result.to_dict()['alternatives']) == 0


def test_zero_window_equal_average_lower_peak():
    result = TrainingResult()
    result.insert(synthetic('2', 5), Fraction(0))
    result.insert(synthetic('2.00000000000000000001', 2), Fraction(0))
    assert result.schedules == [synthetic('2', 5)]
    result.insert(synthetic('2', 3), Fraction(0))
    assert result.schedules == [synthetic('2', 3)]
    result.insert(synthetic('3/2', 4), Fraction(0))
    assert result.schedules == [synthetic('3/2', 4)]


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', '-inf', 'no', '', '1e999', True])
def test_invalid_reward_divisor(value):
    with pytest.raises(ValueError, match='reward_divisor'):
        TrainingReward(AoIEnv([[0, 1], [1, 0]]), value)


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf', '-inf', 'no', '', True])
def test_invalid_average_window(value):
    with pytest.raises(ValueError, match='average_window'):
        parse_average_window(value)


def test_decimal_window_is_exact():
    assert parse_average_window('0.1') == Fraction(1, 10)
    assert parse_average_window('2') == 2
    assert parse_average_window('0') == 0


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
def test_window_does_not_change_optimization(tmp_path, monkeypatch, algorithm):
    module = importlib.import_module('aoi.train')
    models = []
    def capture(env, **kwargs):
        model = make_model(env, **kwargs)
        models.append(model)
        return model
    monkeypatch.setattr(module, 'make_model', capture)
    for window in ('0', '2'):
        train([[0, 1], [1, 0]], tmp_path/f'{window}.json', algorithm=algorithm,
              timesteps=16, average_window=window, max_steps=10,
              validation_every=4, algorithm_options=options(algorithm))
    for x, y in zip(models[0].policy.parameters(), models[1].policy.parameters()):
        assert torch.equal(x, y)


def test_validation_follows_final_optimizer_update(tmp_path, monkeypatch):
    module = importlib.import_module('aoi.train')
    updates = []
    original = module.evaluate_policy
    def record(model, env, **kwargs):
        updates.append(model._n_updates)
        return original(model, env, **kwargs)
    monkeypatch.setattr(module, 'evaluate_policy', record)
    train([[0, 1], [1, 0]], tmp_path/'result.json', timesteps=8,
          validation_every=8, max_steps=10, algorithm_options=options('ppo'))
    assert updates == [0, 1]


def test_default_validation_and_honest_failure():
    callback = ScheduleFrontier(lambda: AoIEnv([[0, 1], [1, 0]]))
    assert callback.every == 25_000
    assert callback.result.to_dict() == {'status': 'evaluation_limit'}


@pytest.mark.parametrize('restricted', [False, True])
def test_training_wrapper_gym_interface(restricted, monkeypatch):
    from stable_baselines3.common.env_checker import check_env
    env = TrainingReward((ConnectedBlockEnv if restricted else AoIEnv)([[0, 1], [1, 0]]))
    assert env.divisor == 2
    if restricted:
        # The generic checker does not know about state-dependent action masks.
        sample = env.action_space.sample
        monkeypatch.setattr(env.action_space, 'sample',
                            lambda: sample(mask=env.unwrapped.action_masks().astype(np.int8)))
    check_env(env)

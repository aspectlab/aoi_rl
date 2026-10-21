import networkx as nx
import numpy as np
import pytest
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from aoi.env import AoIEnv
from aoi.observations import PolicyObservation, progress_mask
from aoi.policies import MaskedActorCriticPolicy, MaskedDQN
from aoi.policy_eval import evaluate_policy
from aoi.train import make_model, TrainingReward


torch.set_num_threads(1)
A = nx.to_numpy_array(nx.path_graph(3))


def dqn(**kwargs):
    env = AoIEnv(A)
    settings = dict(buffer_size=16, batch_size=1, learning_starts=0,
                    learning_rate=0, train_freq=1, target_update_interval=8)
    settings.update(kwargs)
    model = make_model(env, algorithm='dqn', progress_mask=True, seed=42, **settings)
    model._setup_learn(1)
    return env, model


def batch(obs):
    return {k: v[None] for k, v in obs.items()}


def tensors(obs):
    return {k: torch.as_tensor(v) for k, v in obs.items()}


def biased_q(model):
    with torch.no_grad():
        for net in (model.q_net, model.q_net_target):
            for parameter in net.parameters():
                parameter.zero_()
            net.q_net[-1].bias[1] = 1e6  # Illegal, largest raw value.
        model.q_net.q_net[-1].bias[0] = 2
        model.q_net.q_net[-1].bias[4] = 8
        model.q_net_target.q_net[-1].bias[0] = 5
        model.q_net_target.q_net[-1].bias[4] = 3


@pytest.mark.parametrize('done,timeout', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('all_true', [False, True])
def test_actual_standard_dqn_bellman_loss_ignores_illegal_max(done, timeout, all_true):
    env, model = dqn()
    biased_q(model)
    obs = batch(PolicyObservation(env, progress_mask=True).reset()[0])
    assert not obs['mask'][0, 1]
    if all_true:
        obs['mask'][:] = 1
    model.replay_buffer.add(obs, obs, np.array([0]), np.array([1.]),
                            np.array([done]), [{'TimeLimit.truncated': timeout}])
    model.train(1, 1)
    max_next = 1e6 if all_true else 5  # Standard DQN, NOT online argmax's target 3.
    target = 1 + (not done or timeout) * model.gamma * max_next
    expected = torch.nn.functional.smooth_l1_loss(torch.tensor([2.]), torch.tensor([target]))
    assert model.logger.name_to_value['train/loss'] == pytest.approx(expected.item())
    assert MaskedDQN.train is DQN.train
    assert MaskedDQN._on_step is DQN._on_step


def test_dqn_epsilon_zero_one_warmup_and_uniform_exploration():
    env, model = dqn()
    biased_q(model)
    obs = PolicyObservation(env, progress_mask=True).reset()[0]
    mask = obs['mask'].astype(bool)
    model.exploration_rate = 0
    for deterministic in (False, True):
        assert int(model.predict(obs, deterministic=deterministic)[0]) == 4
    model.exploration_rate = 1
    draws = np.array([int(model.predict(obs)[0]) for _ in range(6000)])
    assert mask[draws].all()
    counts = np.bincount(draws, minlength=9)[mask]
    assert np.all(np.abs(counts - 6000 / mask.sum()) < 120)
    for _ in range(100):
        actions, replay_actions = model._sample_action(100)
        assert mask[actions].all()
        np.testing.assert_array_equal(actions, replay_actions)
    assert int(model.predict(obs, deterministic=True)[0]) == 4
    # Batched calls preserve each state's separate support.
    two = {k: np.repeat(v[None], 2, axis=0) for k, v in obs.items()}
    two['mask'][1] = 0
    two['mask'][1, 8] = 1
    for _ in range(30):
        assert model.predict(two)[0][1] == 8


def test_replay_exact_next_masks_align_after_ring_wrap_and_sampling():
    env, model = dqn(learning_rate=1e-4)
    expected = {}

    class Audit(BaseCallback):
        def _on_step(self):
            slot = self.model.replay_buffer.pos
            obs, next_obs = self.model._last_obs, self.locals['new_obs']
            np.testing.assert_array_equal(next_obs['mask'][0], progress_mask(env.model, env.state))
            expected[slot] = ({k: v[0].copy() for k, v in obs.items()},
                              {k: v[0].copy() for k, v in next_obs.items()},
                              int(self.locals['actions'][0]))
            return True

    model.learn(53, callback=Audit())
    buffer = model.replay_buffer
    assert buffer.full
    assert buffer.observations['mask'].dtype.itemsize == 1
    assert buffer.next_observations['mask'].dtype.itemsize == 1
    samples = buffer._get_samples(np.arange(16))
    for slot, (obs, successor, action) in expected.items():
        for k in obs:
            np.testing.assert_array_equal(samples.observations[k][slot], obs[k])
            np.testing.assert_array_equal(samples.next_observations[k][slot], successor[k])
        assert samples.actions[slot].item() == action
        assert obs['mask'][action]
    assert not samples.dones.any()
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())


def test_replay_terminal_observation_keeps_terminal_mask_instead_of_reset_mask():
    class Ends(AoIEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, True, False, info

    env = Ends(A)
    model = make_model(env, algorithm='dqn', progress_mask=True, buffer_size=16,
                       learning_starts=100, train_freq=1, seed=72)
    model.learn(8)
    b = model.replay_buffer
    for k in range(8):
        action = int(b.actions[k, 0, 0])
        terminal_state = env.model.transition(env.model.initial_state, action)
        np.testing.assert_array_equal(b.next_observations['mask'][k, 0],
                                      progress_mask(env.model, terminal_state))
        np.testing.assert_array_equal(b.next_observations['ages'][k, 0], (terminal_state / 6).astype(np.float32))
        assert b.dones[k, 0]
    assert any(not np.array_equal(b.observations['mask'][k], b.next_observations['mask'][k])
               for k in range(8))


def test_ppo_rollout_support_likelihood_entropy_gradient_and_updates():
    env = AoIEnv(A)
    model = make_model(env, progress_mask=True, seed=82, n_steps=32, batch_size=8, n_epochs=2)
    expected = []
    initial = {k: v.clone() for k, v in model.policy.state_dict().items()}

    class Audit(BaseCallback):
        def _on_step(self):
            before = env.model.initial_state if not expected else expected[-1][1]
            mask = progress_mask(env.model, before)
            obs = self.model._last_obs
            np.testing.assert_array_equal(obs['mask'][0], mask)
            action = int(self.locals['actions'][0])
            assert mask[action]
            expected.append((mask, env.state.copy()))
            return True

        def _on_rollout_end(self):
            b = self.model.rollout_buffer
            np.testing.assert_array_equal(b.observations['mask'][:, 0],
                                          [mask for mask, _ in expected[-32:]])
            for data in b.get(8):
                obs = data.observations
                values, logp, entropy = model.policy.evaluate_actions(obs, data.actions.flatten().long())
                assert torch.allclose(logp, data.old_log_prob, atol=2e-6)
                d = model.policy.get_distribution(obs).distribution
                assert (d.probs[~obs['mask'].bool()] == 0).all()
                reference = torch.distributions.Categorical(logits=model.policy.action_net(
                    model.policy.mlp_extractor(model.policy.extract_features(obs))[0]
                ).masked_fill(~obs['mask'].bool(), -torch.inf))
                assert torch.allclose(entropy, reference.entropy(), atol=1e-6)
                assert values.shape == (8, 1)

    model.learn(64, callback=Audit())
    assert any(not torch.equal(initial[k], v) for k, v in model.policy.state_dict().items())
    obs = tensors(batch(PolicyObservation(env, progress_mask=True).reset()[0]))
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
        model.policy.action_net.bias[1] = 1e6
    for _ in range(100):
        action, _, _ = model.policy(obs)
        assert obs['mask'][0, int(action)]
    _, logp, entropy = model.policy.evaluate_actions(obs, torch.tensor([0]))
    assert entropy.item() == pytest.approx(np.log(int(obs['mask'].sum())))
    model.policy.zero_grad()
    (-(logp + entropy).sum()).backward()
    assert model.policy.action_net.bias.grad[1] == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.policy.parameters())


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
def test_unmasked_full_support_and_all_true_mask_equivalence(algorithm):
    class AllTrue(PolicyObservation):
        def observation(self, observation):
            result = super().observation(observation)
            result['mask'][:] = 1
            return result

    settings = (dict(n_steps=16, batch_size=8, n_epochs=1) if algorithm == 'ppo' else
                dict(buffer_size=32, learning_starts=0, batch_size=4, train_freq=1,
                     exploration_initial_eps=0, exploration_final_eps=0))
    ordinary = make_model(AoIEnv(A), algorithm=algorithm, seed=9, **settings)
    obs = ordinary.env.reset()
    if algorithm == 'ppo':
        d = ordinary.policy.get_distribution(torch.as_tensor(obs)).distribution
        assert (d.probs > 0).all()
        with torch.no_grad():
            ordinary.policy.action_net.weight.zero_()
            ordinary.policy.action_net.bias.zero_()
            ordinary.policy.action_net.bias[1] = 100
    else:
        with torch.no_grad():
            for p in ordinary.q_net.parameters():
                p.zero_()
            ordinary.q_net.q_net[-1].bias[1] = 100
    assert int(ordinary.predict(obs, deterministic=True)[0][0]) == 1  # Nonprogress action allowed.
    ordinary = make_model(AoIEnv(A), algorithm=algorithm, seed=9, **settings)
    ordinary.learn(16)
    masked = make_model(AoIEnv(A), algorithm=algorithm, progress_mask=True, seed=9, **settings)
    masked.set_env(TrainingReward(AllTrue(AoIEnv(A), progress_mask=True)))
    masked.learn(16)
    # The all-true calculation reproduces ordinary SB3, including optimization.
    for k, v in ordinary.policy.state_dict().items():
        assert torch.equal(v, masked.policy.state_dict()[k]), k
    network = masked.policy if algorithm == 'ppo' else masked.q_net
    assert network.features_dim == 6


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('masked', [False, True])
def test_validation_uses_model_scaling_and_exact_support(algorithm, masked):
    settings = dict(n_steps=8, batch_size=4) if algorithm == 'ppo' else dict(buffer_size=16)
    model = make_model(AoIEnv(A), algorithm=algorithm, progress_mask=masked,
                       observation_divisor=17, **settings)
    evaluation = AoIEnv(A)
    original = model.predict
    calls = []

    def predict(obs, deterministic=True):
        assert deterministic
        ages = obs['ages'] if masked else obs
        np.testing.assert_array_equal(ages, (evaluation.state / 17).astype(np.float32))
        if masked:
            np.testing.assert_array_equal(obs['mask'], progress_mask(evaluation.model, evaluation.state))
        action, state = original(obs, deterministic=deterministic)
        if masked:
            assert obs['mask'][int(action)]
        calls.append(int(action))
        return action, state

    model.predict = predict
    evaluate_policy(model, evaluation, max_steps=40)
    assert calls

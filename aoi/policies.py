"""SB3 policies that apply connected-block masks throughout learning."""
import math

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


def observation_mask(obs, n):
    """Reconstruct legality from stored one-hot source and holder bits."""
    active = obs[..., -2*n:-n] > 0.5
    holders = obs[..., -n:] > 0.5
    return (torch.eye(n, dtype=torch.bool, device=obs.device)
            | (holders.unsqueeze(-1) & active.unsqueeze(-2))).flatten(-2)


class MaskedQNetwork(QNetwork):
    def forward(self, obs):
        q = super().forward(obs)
        return q.masked_fill(~observation_mask(obs, math.isqrt(self.action_space.n)), -torch.inf)


class MaskedDQNPolicy(DQNPolicy):
    def make_q_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**args).to(self.device)


class MaskedDQN(DQN):
    """Legal warmup/exploration, with masked online and target Q networks."""

    def __init__(self, policy, env, **kwargs):
        if policy == "MlpPolicy":
            policy = MaskedDQNPolicy
        super().__init__(policy, env, **kwargs)
        if not issubclass(self.policy_class, MaskedDQNPolicy):
            raise ValueError("MaskedDQN requires MaskedDQNPolicy.")

    def _random_legal(self, observation):
        obs = np.asarray(observation)
        n = math.isqrt(self.action_space.n)
        active, holders = obs[..., -2*n:-n] > 0.5, obs[..., -n:] > 0.5
        mask = (np.eye(n, dtype=bool)
                | (holders[..., :, None] & active[..., None, :])).reshape(-1, n*n)
        actions = np.array([np.random.choice(np.flatnonzero(row)) for row in mask])
        return actions if obs.ndim > 1 else actions[0]

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if not deterministic and np.random.rand() < self.exploration_rate:
            return self._random_legal(observation), state
        return self.policy.predict(observation, state, episode_start, deterministic)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            actions = self._random_legal(self._last_obs)
            return actions, actions.copy()
        return super()._sample_action(learning_starts, action_noise, n_envs)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    """Use identical masks for PPO sampling, inference, likelihood, and entropy."""

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        actor, critic = features
        return self.mlp_extractor.forward_actor(actor), self.mlp_extractor.forward_critic(critic)

    def _distribution(self, obs, latent):
        logits = self.action_net(latent)
        mask = observation_mask(obs, math.isqrt(self.action_space.n))
        # A finite sentinel prevents undefined entropy gradients from zero * -inf.
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        actor, critic = self._latents(obs)
        distribution = self._distribution(obs, actor)
        actions = distribution.get_actions(deterministic=deterministic)
        return (actions.reshape((-1, *self.action_space.shape)),
                self.value_net(critic), distribution.log_prob(actions))

    def evaluate_actions(self, obs, actions):
        actor, critic = self._latents(obs)
        distribution = self._distribution(obs, actor)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        features = BasePolicy.extract_features(self, obs, self.pi_features_extractor)
        return self._distribution(obs, self.mlp_extractor.forward_actor(features))

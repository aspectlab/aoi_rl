"""Focused SB3 support for exact physical-progress masks."""
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class AgesOnly(BaseFeaturesExtractor):
    """Exclude auxiliary mask metadata from both actor/critic and Q features."""

    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space["ages"].shape[0])

    def forward(self, obs):
        return obs["ages"]


class MaskedQNetwork(QNetwork):
    def forward(self, obs):
        q = super().forward(obs)
        # SB3's standard DQN uses this same forward path for greedy selection
        # and target-network maximization. Replay supplies the NEXT state's
        # exact mask, including terminal observations handled by SB3 itself.
        return q.masked_fill(~obs["mask"].bool(), -torch.inf)


class MaskedDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = AgesOnly
        super().__init__(*args, **kwargs)

    def make_q_net(self):
        args = self._update_features_extractor(self.net_args, features_extractor=None)
        return MaskedQNetwork(**args).to(self.device)


class MaskedDQN(DQN):
    """Uniform legal exploration; unchanged SB3 Bellman loss and target schedule.

    DictReplayBuffer stores current and next masks as one-byte entries, alongside
    the corresponding ages. No exact physical states are duplicated in replay.
    """

    def _random_legal(self, observation):
        mask = np.asarray(observation["mask"], dtype=bool)
        actions = np.array([np.random.choice(np.flatnonzero(row))
                            for row in mask.reshape(-1, self.action_space.n)])
        return actions if mask.ndim > 1 else actions[0]

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
    """Use buffered exact support for sampling, likelihood, entropy and updates."""

    def __init__(self, *args, **kwargs):
        kwargs["features_extractor_class"] = AgesOnly
        super().__init__(*args, **kwargs)

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        actor, critic = features
        return self.mlp_extractor.forward_actor(actor), self.mlp_extractor.forward_critic(critic)

    def _distribution(self, obs, latent):
        logits = self.action_net(latent)
        # A finite sentinel prevents undefined entropy gradients from zero * -inf.
        logits = logits.masked_fill(~obs["mask"].bool(), torch.finfo(logits.dtype).min)
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

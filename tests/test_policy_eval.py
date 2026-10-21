import numpy as np
import pytest
from aoi.env import AoIEnv
from aoi.policy_eval import PolicyResult, evaluate_policy
from aoi.schedules import ScheduleResult, evaluate_schedule


class OldestSource:
    def predict(self, obs, deterministic=True):
        assert deterministic
        return (0 if obs[1] >= obs[0] else 3), None


def test_exact_transient_cycle_and_replay_from_initialization():
    a = [[0, 1], [1, 0]]
    result = evaluate_policy(OldestSource(), AoIEnv(a), max_steps=20)
    assert result.status == 'cycle'
    assert result.transient_actions == ((0, 0),)
    assert result.schedule.actions == ((1, 1), (0, 0))
    assert result.schedule.metrics.period == 2
    assert result.schedule.metrics.continuous_average_aoi == 2
    assert result.schedule.metrics.continuous_peak_aoi == 3
    assert result.to_dict()['transient_length'] == len(result.transient_actions)
    assert evaluate_policy(OldestSource(), AoIEnv(a), max_steps=result.steps_examined) == result
    env = AoIEnv(a)
    for pair in result.transient_actions:
        env.step(env.action_index(pair))
    boundary = env.state_key()
    for pair in result.schedule.actions:
        obs = env.observation()
        assert env.action_pair(OldestSource().predict(obs)[0]) == pair
        env.step(env.action_index(pair))
    assert env.state_key() == boundary
    assert evaluate_schedule(a, result.schedule.actions).metrics == result.schedule.metrics


def test_limit_does_not_export_a_schedule_or_invent_transient():
    class Constant:
        def predict(self, obs, deterministic=True):
            return 0, None
    result = evaluate_policy(Constant(), AoIEnv([[0, 1], [1, 0]]), max_steps=10)
    assert result.to_dict() == {'status': 'evaluation_limit', 'steps_examined': 10}
    assert result.schedule is None
    assert result.transient_actions == ()


def test_float_observation_alias_is_not_integer_state_recurrence():
    class LargeStart(AoIEnv):
        def reset(self, **kwargs):
            super().reset(**kwargs)
            self.state[:] = 2**26
            return self.observation(), self.info()
    class Constant:
        def predict(self, obs, deterministic=True):
            return 0, None
    result = evaluate_policy(Constant(), LargeStart([[0, 1], [1, 0]]), max_steps=5)
    assert result.status == 'evaluation_limit'


def test_full_observed_word_is_preserved_separately_from_replay_period():
    word = ((1, 1), (0, 0)) * 2
    metrics = evaluate_schedule([[0, 1], [1, 0]], word).metrics
    result = PolicyResult(ScheduleResult(word, metrics), ((0, 0),), 5)
    data = result.to_dict()
    assert data['cycle_actions'] == [list(a) for a in word]
    assert len(data['cycle_actions']) == 4
    assert data['period'] == 2

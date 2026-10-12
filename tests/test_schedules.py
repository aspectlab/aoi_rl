from fractions import Fraction
import numpy as np
import pytest
from aoi.schedules import EvaluationLimit, evaluate_schedule


@pytest.mark.parametrize('n', [2, 3, 5, 7])
def test_complete_graph_closed_form(n):
    a = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
    word = tuple((j, j) for j in range(n))
    result = evaluate_schedule(a, word)
    assert result.metrics.continuous_average_aoi == Fraction(n + 2, 2)
    assert result.metrics.continuous_peak_aoi == n + 1
    assert result.metrics.period == n
    assert evaluate_schedule(a, word * 3) == result
    for phase in range(n):
        assert evaluate_schedule(a, word[phase:] + word[:phase]).metrics == result.metrics


def test_hand_calculated_relay_schedule():
    a = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    word = [(0, 0), (1, 0), (1, 1), (2, 2), (1, 2)]
    result = evaluate_schedule(a, word)
    assert result.metrics.continuous_average_aoi == Fraction(23, 6)
    assert result.metrics.continuous_peak_aoi == 7
    assert result.to_dict()['continuous_average_aoi'] == '23/6'


def test_limit_is_not_finite_metric():
    with pytest.raises(EvaluationLimit):
        evaluate_schedule([[0, 1], [1, 0]], [(0, 0)], max_repetitions=5)


@pytest.mark.parametrize('word', [[], [[0, 2]], [[0.0, 0]], [[True, 0]], [[0]], None])
def test_invalid_words(word):
    with pytest.raises(ValueError):
        evaluate_schedule([[0, 1], [1, 0]], word)


def test_recurrence_requires_matching_word_phase():
    result = evaluate_schedule([[0, 1], [1, 0]], [(0, 0), (1, 1), (0, 0), (1, 1), (1, 1)])
    assert result.metrics.period == 5
    assert result.metrics.continuous_average_aoi == Fraction(21, 10)
    assert result.metrics.continuous_peak_aoi == 4


def test_refreshing_word_washes_out_different_initial_states():
    from aoi.env import AoIModel
    model = AoIModel([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    word = [(0, 0), (1, 0), (1, 1), (2, 2), (1, 2)]
    states = [model.initial_state.copy(), np.zeros(6, dtype=np.int64),
              np.full(6, 10000, dtype=np.int64)]
    for _ in range(20):
        for pair in word:
            states = [model.transition(x, model.action_index(pair)) for x in states]
    for state in states[1:]:
        np.testing.assert_array_equal(state, states[0])

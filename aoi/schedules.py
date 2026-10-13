"""Exact recurrent evaluation of externally repeated transmission words."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .env import AoIModel


class EvaluationLimit(RuntimeError):
    """No exact recurrence was established within the requested budget."""


def positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_word(model: AoIModel, actions) -> tuple[tuple[int, int], ...]:
    try:
        word = tuple(model.action_pair(model.action_index(pair)) for pair in actions)
    except (TypeError, ValueError) as exc:
        raise ValueError("Schedule must be a list of integer [transmitter, process] pairs.") from exc
    if not word:
        raise ValueError("Schedule must be nonempty.")
    return word


def primitive_word(word):
    """Remove whole-word repetitions without changing the initial phase."""
    for length in range(1, len(word) + 1):
        if len(word) % length == 0 and word == word[:length] * (len(word) // length):
            return word[:length]
    raise ValueError("Schedule must be nonempty.")


@dataclass(frozen=True)
class CycleMetrics:
    period: int
    continuous_average_aoi: Fraction
    continuous_peak_aoi: int

    def to_dict(self):
        return {"period": self.period,
                "continuous_average_aoi": str(self.continuous_average_aoi),
                "continuous_peak_aoi": self.continuous_peak_aoi}


def cycle_metrics(total_age: int, peak_age: int, period: int, n: int) -> CycleMetrics:
    """Integrate a closed integer-state cycle; peak is a supremum, not a mean."""
    # Each unit slot contributes boundary age + 1/2 to its integral and approaches
    # boundary age + 1 before reception. On a closed cycle, pre/post-step boundary
    # samples differ only by a cyclic shift; these are not boundary-only metrics.
    return CycleMetrics(period, Fraction(total_age, period * n * (n - 1))
                        + Fraction(1, 2), peak_age + 1)


@dataclass(frozen=True)
class ScheduleResult:
    actions: tuple[tuple[int, int], ...]
    metrics: CycleMetrics

    def to_dict(self):
        return {"status": "cycle", "cycle_actions": [list(a) for a in self.actions], **self.metrics.to_dict()}


def evaluate_schedule(adjacency, actions, *, max_repetitions: int = 1000) -> ScheduleResult:
    """Repeat a fixed word until the full integer state repeats at word phase zero.

    Always uses unrestricted physical dynamics. The limit is a resource budget;
    exceeding it does not establish that the word has unbounded ages.

    Parameters
    ----------
    adjacency : array-like, shape (N, N)
        Graph satisfying AoIEnv's adjacency requirements.
    actions : iterable of (int, int)
        Nonempty word of (transmitter, process) pairs, repeated from initialization.
    max_repetitions : int
        Positive budget in repetitions of the primitive word.

    Returns
    -------
    ScheduleResult
        Primitive word (same phase) and exact recurrent continuous metrics,
        excluding the transient. The reported period can span multiple copies.

    Raises
    ------
    EvaluationLimit
        No matching word-boundary state was found within the budget.
    """
    positive_integer(max_repetitions, "max_repetitions")
    model = AoIModel(adjacency)
    word = primitive_word(validate_word(model, actions))
    indices = tuple(model.action_index(a) for a in word)
    state = model.initial_state.copy()
    seen = {}
    # A fixed word is an external clock: physical equality is sufficient only
    # at matching word phase. Cached updates can persist across repetitions,
    # so the word's length alone does not establish physical state closure.
    for repetition in range(max_repetitions + 1):
        key = tuple(map(int, state))
        if key in seen:
            repetitions = repetition - seen[key]
            break
        if repetition == max_repetitions:
            raise EvaluationLimit("No repeated word-boundary state within the repetition limit.")
        seen[key] = repetition
        for action in indices:
            state = model.transition(state, action)
    boundary = state.copy()
    total, peak = 0, 0
    for _ in range(repetitions):
        for action in indices:
            state = model.transition(state, action)
            total += sum(map(int, state))
            peak = max(peak, int(state.max()))
    if not (state == boundary).all():
        raise RuntimeError("Recurrent word evaluation failed exact closure.")
    return ScheduleResult(word, cycle_metrics(total, peak, repetitions * len(word), model.n))

"""Full-state recurrence of a frozen deterministic policy, from initialization."""
from dataclasses import dataclass

import numpy as np

from .blocks import ConnectedBlockEnv, is_cyclic_block_schedule
from .schedules import (EvaluationLimit, ScheduleResult, cycle_metrics,
                        evaluate_schedule, positive_integer)


@dataclass(frozen=True)
class PolicyResult:
    schedule: ScheduleResult | None
    transient_actions: tuple[tuple[int, int], ...] = ()
    steps_examined: int = 0

    @property
    def status(self):
        return "cycle" if self.schedule is not None else "evaluation_limit"

    def to_dict(self):
        if self.schedule is None:
            return {"status": self.status, "steps_examined": self.steps_examined}
        return self.schedule.to_dict() | {
            "transient_actions": [list(a) for a in self.transient_actions],
            "transient_length": len(self.transient_actions)}


def evaluate_policy(model, env, *, max_steps=50_000) -> PolicyResult:
    """Record the exact transient and controlled cycle of a stationary MLP policy.

    The returned action word preserves the complete controlled-state recurrence,
    including its phase. Its length can differ from the independently evaluated
    periodic word's physical period; the latter is stored in the metrics.

    Parameters
    ----------
    model : object with predict(obs, deterministic=True)
        Frozen, stationary, memoryless policy; hidden recurrent state is unsupported.
    env : AoIEnv or ConnectedBlockEnv
        Continuing evaluation environment; this call resets and advances it.
    max_steps : int
        Positive controlled-trajectory budget, also used as replay repetition budget.

    Returns
    -------
    PolicyResult
        Verified cycle and transient, or status ``evaluation_limit`` if either
        recurrence search exhausts its budget. A limit is not proof of divergence;
        a verified cycle certifies this trajectory, not global optimality.
    """
    positive_integer(max_steps, "max_steps")
    obs, _ = env.reset()
    seen, actions, totals, peaks = {}, [], [], []
    for t in range(max_steps + 1):
        # A fixed policy supplies the next action, so no external word phase is
        # needed. All physical ages AND block memory must recur: equal rewards,
        # actions, or rounded float observations do not determine the same future.
        key = env.state_key()
        if key in seen:
            start = seen[key]
            word = tuple(actions[start:])
            measured = cycle_metrics(sum(totals[start:]), max(peaks[start:]), len(word), env.n)
            try:
                replay = evaluate_schedule(env.model.adjacency, word, max_repetitions=max_steps)
            except EvaluationLimit:
                return PolicyResult(None, steps_examined=t)
            if (replay.metrics.continuous_average_aoi != measured.continuous_average_aoi
                    or replay.metrics.continuous_peak_aoi != measured.continuous_peak_aoi):
                raise RuntimeError("Exported word disagrees with policy-cycle metrics.")
            if (isinstance(env, ConnectedBlockEnv)
                    and not is_cyclic_block_schedule(env.model.adjacency, word)):
                raise RuntimeError("Exported word violates cyclic block legality.")
            return PolicyResult(ScheduleResult(word, replay.metrics), tuple(actions[:start]), t)
        if t == max_steps:
            break
        seen[key] = t
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action).item()
        actions.append(env.action_pair(action))
        totals.append(sum(map(int, env.state)))
        peaks.append(int(env.state.max()))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            raise RuntimeError("Policy evaluation requires a continuing environment.")
    return PolicyResult(None, steps_examined=max_steps)

"""Scaled policy ages and exact auxiliary action support; no physical changes."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces


def parse_divisor(value, name):
    """Validate either independent numerical divisor."""
    try:
        divisor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(divisor) or divisor <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return divisor


def default_divisor(model):
    # Initialization contains every nonlocal shortest-path distance, whose
    # maximum is the diameter. No MCDS calculation enters numerical scaling.
    return model.n * int(model.initial_state.max())


def progress_mask(model, state):
    """Legal N² actions from the graph and exact integer (monitor, source) ages.

    Sources are always legal. A relay i,j is useful iff some neighbor m != j
    has A[i,j] < A[m,j]. Both ages advance by one in the physical transition,
    so this is precisely strict improvement relative to passive aging.
    """
    state = np.asarray(state)
    if (state.shape != model.initial_state.shape or state.dtype.kind not in "iu"
            or np.any(state < 0)):
        raise ValueError("State must be a nonnegative integer age vector.")
    local = np.eye(model.n, dtype=bool)
    ages = np.zeros((model.n, model.n), dtype=state.dtype)
    ages[~local] = state
    # Axis order: transmitter, receiver, source. Never compare a local receiver.
    useful = ((ages[:, None, :] < ages[None, :, :])
              & model.adjacency[:, :, None] & ~local[None, :, :])
    return (useful.any(axis=1) | local).reshape(-1)


class PolicyObservation(gym.ObservationWrapper):
    """One physical-age representation, with optional buffered mask metadata.

    Mask bits travel in an SB3 Dict buffer but never enter neural features.
    The underlying AoIEnv remains unrestricted and returns its original reward.
    """

    def __init__(self, env, divisor=None, *, progress_mask=False):
        super().__init__(env)
        self.divisor = (default_divisor(env.unwrapped.model) if divisor is None
                        else parse_divisor(divisor, "observation_divisor"))
        self.progress_mask = progress_mask
        if progress_mask:
            self.observation_space = spaces.Dict({
                "ages": env.observation_space,
                "mask": spaces.MultiBinary(int(env.action_space.n))})

    def observation(self, observation):
        # Divide in float64 before the final float32 conversion so finite
        # divisors outside float32's range are not rounded to zero or infinity.
        ages = (self.unwrapped.state.astype(np.float64) / self.divisor).astype(np.float32)
        if not self.progress_mask:
            return ages
        return {"ages": ages, "mask": progress_mask(
            self.unwrapped.model, self.unwrapped.state).astype(np.int8)}

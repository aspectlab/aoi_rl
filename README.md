# Age of Information scheduling

**Adjacency matrix → Algorithm 1 / PPO / DQN → periodic schedule → exact AoI.**
Learned schedules also include the deterministic transient from initialization.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

For tests, install `pip install '.[dev]'` and run `pytest -q`. Dependencies are
NumPy, NetworkX, Gymnasium, Stable-Baselines3, and PyTorch. SB3 is pinned to the
version supported by the custom action masks.

## Model and graph input

Each node is the source of one process. Each slot contains one action
`(transmitter, process)`. A source samples fresh information at the start of the
slot; all one-hop neighbors receive it at the next boundary with age 1. A relay
broadcasts its pre-slot cached copy. Receivers keep the fresher information.
Every nonlocal age increases by one per slot, with no forwarding within a slot.
Local ages are zero and excluded from metrics.

Supply a NumPy `.npy` file containing a finite binary square adjacency matrix.
It must be symmetric, have zero diagonal, contain at least two nodes, and be
connected. Weighted matrices are rejected. Node labels are the unchanged matrix
indices `0, ..., N-1`.

```python
import numpy as np
np.save("graph.npy", [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
```

The integer state contains all `(monitor, process)` ages with unequal indices,
in monitor-major order. Initialization assigns graph-distance ages. This is a
computational initialization, distinct from undefined ages before reception.
A learned policy can reach different attractors from different initial states.

The physical model and Algorithm 1 follow Farazi, Klein, and Brown,
*Fundamental Bounds on the Age of Information in Multi-Hop Global Status Update
Networks*, Journal of Communications and Networks, 2019.

## Algorithm 1

```bash
python -m aoi algorithm1 graph.npy --output algorithm1.json
```

For each source, choose a minimum connected dominating set (MCDS) containing it.
If none exists, augment an MCDS with that source. Emit one transmission per vertex
in a DFS of the induced graph rooted at the source. Sources are processed in
label order, MCDS ties are resolved lexicographically, and DFS visits neighbors
in ascending order.

The word length is `T* = N*gamma_c + |L|`, where `gamma_c` is the minimum connected
domination number and `L` consists of vertices in no MCDS. MCDS enumeration is
exact and can take exponential time. Only Algorithm 1 performs this search.

## PPO and DQN

```bash
python -m aoi train graph.npy --algorithm ppo --output ppo.json
python -m aoi train graph.npy --algorithm dqn --output dqn.json
python -m aoi train graph.npy --algorithm ppo --connected-block --output ppo-block.json
python -m aoi train graph.npy --algorithm dqn --connected-block --output dqn-block.json
```

Training saves one JSON file: the best verified schedule, its exact metrics, and
its transient under deterministic evaluation. The neural network is not saved.
A run directory is unnecessary. Existing output files are never replaced.

The environment reward is exactly negative **unscaled post-transmission average
age** over all `N*(N-1)` ordered nonlocal pairs. Peak age does not enter the reward.
PPO and DQN optimize discounted returns from this reward; they do not directly
solve the infinite-horizon average-reward optimization problem.

The process is genuinely continuing: every step returns `terminated=False` and
`truncated=False`. Physical state and block memory persist across PPO rollout
boundaries and DQN updates. Evaluation uses separate environments and does not
reset the live training trajectory.

Defaults are one million requested training steps, discount `0.999`, CPU, and
SB3 MLP settings. PPO can round the budget up to a complete rollout. These are
numerical choices, not model definitions. Use `--timesteps`, `--seed`, `--gamma`,
and `--device` to adjust them.

Every 100,000 steps, and once after the final optimizer update, a frozen policy
is evaluated deterministically from initialization. `--validation-every` changes
this cadence; zero requests final validation only. `--eval-steps` sets the
recurrence search budget (default 50,000). Verified schedules are compared by
exact continuous average AoI; equal averages retain the first result. Only the
best result is retained. Its transient and cycle come from the same frozen
policy evaluation, not from the changing training trajectory.

A finite run need not discover a recurrent schedule. If none is verified,
`result.json` contains `status: "evaluation_limit"` and `steps_examined`, with no
invented schedule or recurrent metrics. This does not establish divergence.

## Action spaces and connected blocks

All `N*N` action indices are present in both modes: pair `(i, j)` has index
`i*N + j`. Unrestricted policies can select any pair, including redundant relays.
Their observations are unnormalized `float32` ages; physical state stays integer.

Connected-block policies additionally observe an active-source one-hot vector
and current-update holder bits. Every source transmission starts a fresh block,
even for the same source. A relay must transmit the active process and already
hold that block's update. Its neighbors become holders. Repeated relays remain
legal; branches can follow any earlier informed transmitter. Switching sources
discards the old block memory.

PPO masks sampling, deterministic prediction, log probabilities, and entropy.
DQN masks warmup, exploration, greedy prediction, online Q values, and successor
target maxima. Memory is stored with observations and included in recurrence
keys. Connected-block search is a heuristic restriction, not proved lossless.

## Schedules, transients, and exact metrics

```bash
python -m aoi evaluate graph.npy ppo.json
```

A successful learned result has this schema (a two-node example):

```json
{
  "status": "cycle",
  "cycle_actions": [[1, 1], [0, 0]],
  "period": 2,
  "continuous_average_aoi": "2",
  "continuous_peak_aoi": 3,
  "transient_actions": [[0, 0]],
  "transient_length": 1
}
```

Pairs are zero-based. The transient begins at shortest-path ages and empty block
memory. `transient_length` is its number of actions. `cycle_actions` preserves
the entire observed controlled-state cycle in order and can then repeat forever.
For learned results its length is the controlled-policy recurrence period.

`period` is the exact physical recurrent period found by independent unrestricted
fixed-word replay, after removing exact repeated copies of a shorter word. It
can differ from the length of the observed controlled-policy cycle. The full
learned action word is retained to preserve its exact ordering and memory
closure. No rotation is applied.

Algorithm 1 and explicit-word evaluation omit the two transient fields. They
construct/evaluate a periodic word, not a learned-policy startup trajectory.
A hand-written input needs only `{"cycle_actions": [[0, 0], [1, 1]]}` for a
suitable graph. Evaluation reads the cycle, ignores any learned transient, and
recomputes the metrics rather than trusting supplied values.

Policy recurrence is detected from the first boundary using the complete integer
state, plus active process and holder memory when restricted. Repeated float
observations, actions, or rewards do not establish recurrence. Any detected word
is independently replayed unrestricted and must agree exactly on both metrics.
Only stationary memoryless policies are supported by this evaluator.

Fixed-word evaluation checks full integer states at matching word phase and
verifies exact closure. If the cycle spans multiple word repetitions, all of
them contribute to its period. `--max-repetitions` controls this evaluator's
budget. Exhaustion returns `{"status": "evaluation_limit"}` without classifying
the word as bounded or unbounded.

For a verified recurrent period `P` and `M=N*(N-1)` ages:

- Continuous average AoI is `sum(all recurrent ages)/(P*M) + 1/2`, computed with
  `Fraction` and serialized as an exact rational string.
- Continuous supremum peak AoI is `max(all recurrent ages) + 1`, an integer.

Ages rise linearly between boundaries. Peak is the supremum before receptions,
not mean instantaneous peak. These metrics exclude the learned transient.
Recurrence establishes the observed trajectory, not global optimality or
convergence from every initialization. Commands exit with code 2 when no verified
schedule is produced or an input is invalid.

## Python API

```python
from aoi import AoIEnv, ConnectedBlockEnv, evaluate_schedule
from aoi.algorithm1 import algorithm1_schedule
from aoi import evaluate_policy
from aoi.train import train

construction = algorithm1_schedule(adjacency)
result = evaluate_schedule(adjacency, construction.actions)
print(result.metrics.continuous_average_aoi)  # fractions.Fraction

selected = train(adjacency, "result.json", algorithm="dqn", connected_block=True)
```

`AoIEnv.model.transition(state, action)` is the pure integer transition.
`evaluate_policy(model, env, max_steps=50000)` evaluates an in-memory frozen model
in a separate environment. `train` accepts `algorithm_options` for SB3 numerical
settings. It returns the selected evaluation and writes the same result as JSON.

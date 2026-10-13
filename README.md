# Age of Information scheduling

**Graph edges → PPO / DQN or Algorithm 1 → self-contained JSON → exact AoI.**
Training retains the best-average word and useful lower-peak alternatives.

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

## Architecture

The physical model and exact evaluators are independent of the RL algorithms:
they can simulate and verify externally supplied words or the constructive
baseline. Training uses the same physical dynamics and delegates verification
to these evaluators.

| Module | Responsibility |
| --- | --- |
| `env.py` | Physical AoI simulation and unrestricted environment. |
| `blocks.py` | Connected-block restrictions and their explicit memory. |
| `policies.py` | PPO/DQN integration and action masking throughout learning. |
| `train.py` | Training, isolated validation, and discovered-frontier retention. |
| `schedules.py`, `policy_eval.py` | Exact fixed-word and deterministic-policy recurrence evaluation. |
| `algorithm1.py` | Constructive baseline using minimum connected dominating sets. |
| `graphs.py` | External graph input/output with label preservation. |
| `__main__.py` | Command-line interface. |

## Model and graph input

Each node is the source of one process. Each slot contains one action
`(transmitter, process)`. A source samples fresh information at the start of the
slot; all one-hop neighbors receive it at the next boundary with age 1. A relay
broadcasts its pre-slot cached copy. Receivers keep the fresher information.
Every nonlocal age increases by one per slot, with no forwarding within a slot.
Local ages are zero and excluded from metrics.

Use exactly one graph source for `train` and `algorithm1`: `--edges` or `--graph`.
Inline edges are comma-separated undirected `u-v` pairs:

```bash
python -m aoi train --edges "0-1,1-2,2-3,3-4,4-5" --algorithm ppo --output p6.json
```

Whitespace around endpoints and separators is allowed. Inline input infers
`N = 1 + max(endpoint)`. Vertices are labeled `0, ..., N-1`; missing labels
create isolated vertices and are rejected. Labels are never renumbered.

Alternatively, save a standalone graph as `graph.json`:

```json
{
  "num_nodes": 4,
  "edges": [[0, 1], [1, 2], [2, 3]]
}
```

These are the only permitted graph fields. `num_nodes` must be an integer at
least 2. Endpoints must be integers in `0, ..., num_nodes-1`, not Booleans.
Graphs must be connected, undirected, and simple: self-loops, duplicate edges
(including reversed duplicates), invalid endpoints, and malformed inputs are
rejected. JSON input uses its explicit node count. Exported edges always have
smaller endpoint first and are sorted lexicographically. This canonicalizes
the edge representation while preserving the labeled graph; differently labeled
isomorphic graphs remain distinct.

The integer state contains all `(monitor, process)` ages with unequal indices,
in monitor-major order. Initialization assigns graph-distance ages. This is a
computational initialization, distinct from undefined ages before reception.
A learned policy can reach different attractors from different initial states.

The physical model and Algorithm 1 follow Farazi, Klein, and Brown,
*Fundamental Bounds on the Age of Information in Multi-Hop Global Status Update
Networks*, Journal of Communications and Networks, 2019.

## Algorithm 1

```bash
python -m aoi algorithm1 --edges "0-1,1-2,2-3" --output algorithm1.json
python -m aoi algorithm1 --graph graph.json --output from-file.json
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
python -m aoi train --graph graph.json --algorithm ppo --output ppo.json
python -m aoi train --graph graph.json --algorithm dqn --output dqn.json
python -m aoi train --graph graph.json --algorithm ppo --connected-block --output ppo-block.json
python -m aoi train --graph graph.json --algorithm dqn --connected-block --output dqn-block.json
```

Training saves one JSON file containing the graph, verified periodic words,
and their exact metrics. Transient histories and neural networks are not saved.
A run directory is unnecessary. Existing output files are never replaced.

The environment reward is exactly negative **unscaled post-transmission average
age** over all `N*(N-1)` ordered nonlocal pairs. Peak age does not enter the reward.
Only the PPO/DQN training interface divides this raw reward by
`S(G) = N*diameter(G)`, computed using NetworkX without MCDS or `T*` search.
This is an empirical numerical-conditioning heuristic, not a mathematical
requirement or a proven optimal scale. Observations and physical dynamics are
unchanged; deterministic evaluation and exact metrics remain unscaled.

Use `--reward-divisor 26` to override the divisor or `--reward-divisor 1` to
train with raw rewards. Explicit divisors must be positive and finite. There is
no clipping, normalization, or nonlinear shaping. PPO and DQN optimize discounted
returns from the scaled reward; they do not directly solve the infinite-horizon
average-reward optimization problem.

The process is genuinely continuing: every step returns `terminated=False` and
`truncated=False`. Physical state and block memory persist across PPO rollout
boundaries and DQN updates. Evaluation uses separate environments and does not
reset the live training trajectory.

Defaults are one million requested training steps, discount `0.999`, CPU, and
SB3 MLP settings. PPO can round the budget up to a complete rollout. These are
numerical choices, not model definitions. Use `--timesteps`, `--seed`, `--gamma`,
and `--device` to adjust them.

Every 25,000 steps, and once after the final optimizer update, a frozen policy
is evaluated deterministically from initialization. `--validation-every` changes
this cadence; zero requests final validation only. `--eval-steps` sets the
recurrence search budget (default 50,000). Validation preserves training RNG state.

The best discovered continuous average AoI is `A*`. The retained frontier
contains exactly the nondominated discovered metric points with `A <= A* + W`.
`--average-window` sets `W` (default `1`), a nonnegative finite value parsed as an
exact rational. The boundary is inclusive. For example, `--average-window 0.5`
allows half an age unit; `--average-window 0` keeps only the best average,
preferring the lowest peak among equal averages.

A point dominates another if its average and peak are both no larger and at
least one is strictly smaller. Thus additional retained words offer strictly
lower peaks at higher averages. Identical metric points keep the first word.
Comparisons use rational averages and integer peaks. As `A*` improves, dominated
and out-of-window words are pruned in memory. Only this frontier is retained;
there are no validation logs or historical discoveries. The window affects
reporting only: the training objective remains average AoI.

A finite run need not discover a recurrent schedule. If none is verified,
`result.json` contains only the top-level `graph` and `"status": "evaluation_limit"`,
with no invented schedule or recurrent metrics. This does not establish divergence.

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

## Periodic schedules and exact metrics

```bash
python -m aoi evaluate ppo.json
python -m aoi evaluate ppo.json --schedule-index 1
```

A successful learned result has this schema (a two-node example):

```json
{
  "graph": {"num_nodes": 2, "edges": [[0, 1]]},
  "status": "cycle",
  "best_average": {
    "cycle_actions": [[1, 1], [0, 0]],
    "period": 2,
    "continuous_average_aoi": "2",
    "continuous_peak_aoi": 3
  },
  "alternatives": []
}
```

`best_average` is always present on success. `alternatives` contains any other
retained words, ordered by increasing average (then peak). Each entry has the
same four fields. Index 0 (the default) selects `best_average`; indices 1, 2, ...
select alternatives. Negative or out-of-range indices are rejected.

Pairs are zero-based. `cycle_actions` preserves the entire observed
controlled-state cycle in chronological order and can then repeat forever.
For learned results its length is the controlled-policy recurrence period.

`period` is the exact physical recurrent period found by independent unrestricted
fixed-word replay, after removing exact repeated copies of a shorter word. It
can differ from the length of the observed controlled-policy cycle. The full
learned action word is retained to preserve its exact ordering and memory
closure. No rotation is applied.

Every exported result embeds its graph exactly once at the top level.
Algorithm 1 and explicit-word evaluation produce a single schedule with `graph`, `status`,
`cycle_actions`, and the three metric fields, without a frontier wrapper.
A hand-written input needs only `{"cycle_actions": [[0, 0], [1, 1]]}` for a
suitable graph, supplied explicitly:

```bash
python -m aoi evaluate schedule.json --graph graph.json
python -m aoi evaluate schedule.json --edges "0-1"
```

Self-contained results require no external graph. If one is supplied, its labeled
adjacency matrix must match the embedded graph exactly; conflicting graphs are
rejected. `--edges` and `--graph` are mutually exclusive even for evaluation.
Single schedules accept only index 0. A failure result has no schedule to replay
and produces a clear input error. Evaluation accepts these formats and training JSON, and always
recomputes metrics through the same independent unrestricted fixed-word evaluator
rather than trusting supplied values.

Policy recurrence is detected from the first boundary using the complete integer
state, plus active process and holder memory when restricted. Repeated float
observations, actions, or rewards do not establish recurrence. Any detected word
is independently replayed unrestricted and must agree exactly on both metrics.
Only stationary memoryless policies are supported by this evaluator.

Fixed-word evaluation checks full integer states at matching word phase and
verifies exact closure. If the cycle spans multiple word repetitions, all of
them contribute to its period. `--max-repetitions` controls this evaluator's
budget. Exhaustion returns the graph and `"status": "evaluation_limit"` without
classifying the word as bounded or unbounded. Use `--output evaluated.json` to
save a self-contained evaluation result; otherwise it is printed to stdout.

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

The Python APIs continue to accept NumPy adjacency matrices.

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
settings, `reward_divisor=None` for automatic scaling (or a positive override),
`average_window="1"` for the exact window, and `validation_every=25000`.
It returns a `TrainingResult`: `schedules` holds the sorted frontier of
`ScheduleResult` objects and `to_dict()` gives the saved JSON schema.
`make_model` applies the same training reward wrapper. The base `AoIEnv` and
`ConnectedBlockEnv` always return raw rewards.

# Age of Information scheduling

This toolkit constructs and evaluates transmission schedules that keep status
information fresh across a network. It learns schedules with PPO or DQN and
provides Algorithm 1 from Farazi, Klein, and Brown (2019) as a constructive
baseline. Each verified periodic schedule is saved with its graph and exact
continuous average and peak Age of Information (AoI).

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

For tests, install `pip install '.[dev]'` and run `python -m pytest -q`. Dependencies are
NumPy, NetworkX, Gymnasium, Stable-Baselines3 (SB3), and PyTorch. SB3 is pinned to
the version supported by the custom action masks.

## Model and graph input

Each node generates status updates for one process and monitors every other
process. AoI is the time elapsed since the generation of the freshest update
available at a monitor. The scheduling objective is to minimize average AoI
across all ordered pairs of distinct nodes.

Each unit-length slot contains one action `(transmitter, process)`. When the
transmitter is the source, it samples fresh information at the start of the slot;
all one-hop neighbors receive it at the next boundary with age 1. A relay
broadcasts its pre-slot cached copy. Receptions are reliable, and receivers keep
the fresher information. Between receptions, every nonlocal age increases by one
per slot. There is no forwarding within a slot. Local ages are zero and excluded
from metrics.

The integer state stores `(monitor, process)` ages with unequal indices in
monitor-major order. Initialization assigns graph-distance ages as a
computational starting state. A learned policy can reach different recurrent
trajectories from different initial states.

Use exactly one graph source for `train` and `algorithm1`: `--edges` or `--graph`.
Inline edges are comma-separated undirected `u-v` pairs:

```bash
python -m aoi train --edges "0-1,1-2,2-3,3-4,4-5" --output p6.json
```

Whitespace around endpoints and separators is allowed. Inline input infers
`N = 1 + max(endpoint)`. Vertices are labeled `0, ..., N-1`; missing labels
create isolated vertices and are rejected. Labels are preserved.

Alternatively, save a graph as `graph.json`:

```json
{
  "graph": {"num_nodes": 4, "edges": [[0, 1], [1, 2], [2, 3]]}
}
```

The inner `graph` has exactly these two fields. `num_nodes` must be an integer at
least 2. Endpoints must be integers in `0, ..., num_nodes-1`, not Booleans.
Graphs must be connected, undirected, and simple: self-loops, duplicate edges
(including reversed duplicates), invalid endpoints, and malformed inputs are
rejected. JSON input uses its explicit node count. Exported edges have the
smaller endpoint first and are sorted lexicographically, preserving node labels.
Differently labeled isomorphic graphs remain distinct inputs.

`--graph` also accepts any toolkit result, reading only its graph. Each output
file must be new; existing paths, including the input path, are rejected.

## Algorithm 1

```bash
python -m aoi algorithm1 --edges "0-1,1-2,2-3" --output algorithm1.json
python -m aoi algorithm1 --graph graph.json --output from-file.json
```

The physical model and Algorithm 1 follow Farazi, Klein, and Brown,
[*Fundamental Bounds on the Age of Information in Multi-Hop Global Status Update
Networks* (2019)](https://doi.org/10.1109/JCN.2019.000038).

A connected dominating set induces a connected subgraph and contains or neighbors
every vertex. For each source, Algorithm 1 chooses a minimum connected dominating
set (MCDS) containing it. If none contains it, the source is added to an MCDS.
Each selected vertex transmits once, in depth-first search (DFS) order starting
at the source. Sources are processed in label order, MCDS ties are resolved
lexicographically, and DFS visits neighbors in ascending order.

The resulting action sequence, or *word*, repeats with length
`T* = N*gamma_c + |L|`, where `gamma_c` is the minimum connected domination number
and `L` consists of vertices in no MCDS. Exact MCDS enumeration can take
exponential time; it is used only by Algorithm 1.

Algorithm 1 reports two exact average-age bounds from the paper:

- Theorem 4, equation (15): `average_lower_tstar_periodic = T*/2 + d_bar`,
  where `d_bar` is the exact mean shortest-path distance over ordered distinct pairs.
  This lower bound applies to T*-periodic schedules updating all statuses.
- Theorem 5, equation (16): `average_upper_algorithm1 = T*/2 + gamma_c + |L|/N`,
  an upper bound on the average achieved by Algorithm 1.

Neither is a universal bound or optimality certificate for arbitrary-period
PPO/DQN schedules. Achieved metrics are computed by independent exact schedule
replay. For the six-node path P6, the bounds are `46/3` and `52/3`, while the
constructed 26-transmission word achieves average `232/15` and supremum peak `31`.
For P7, the constructed word has period `37`, average `899/42`, and supremum peak `43`.

The output contains `graph`, `bounds`, and a one-entry `schedules` array. Each
bound has an exact rational string and a matching `_decimal` JSON number rounded
to six places. To train on the same graph:

```bash
python -m aoi algorithm1 --edges "0-1,1-2,2-3,3-4,4-5" --output alg1.json
python -m aoi train --graph alg1.json --progress-mask --seed 0 --output ppo_seed0.json
```

## PPO and DQN

Proximal Policy Optimization (PPO, the default) learns a policy and value function
from rollouts. Deep Q-Network (DQN) learns action values from replayed transitions.
Both use SB3 multilayer perceptrons and allow all transmission actions by default.

```bash
python -m aoi train --graph graph.json --output ppo.json
python -m aoi train --graph graph.json --algorithm dqn --output dqn.json
python -m aoi train --graph graph.json --progress-mask --output ppo-masked.json
python -m aoi train --graph graph.json --algorithm dqn --progress-mask --output dqn-masked.json
```

Training saves the graph, verified periodic schedules, exact metrics, and the
algorithm, scaling, and masking settings in one JSON file. Neural networks and
transient histories are not saved.

The environment reward is negative **unscaled post-slot average age** over all
`N*(N-1)` ordered nonlocal pairs. Peak age does not enter the reward. For training,
the reward and the `N*(N-1)` age features are each divided by `N*diameter(G)` by
default. These independent divisors are empirical conditioning choices. The
physical state and exact evaluators use integer ages; neural networks receive
scaled floating-point observations. Deterministic policy validation uses the
same observation divisor as training.

```bash
python -m aoi train --graph graph.json --observation-divisor 1 --output raw-ages.json
python -m aoi train --graph graph.json --reward-divisor 1 --observation-divisor 56 --output custom.json
```

Both flags accept positive, finite values; `1` disables the corresponding scaling.
There is no clipping, logarithm, running normalization, or standardization.
PPO and DQN optimize discounted returns from scaled rewards; they do not directly
solve the infinite-horizon average-reward optimization problem.

The environment is continuing: steps return `terminated=False` and
`truncated=False`, and physical state persists across rollouts and updates.
Validation runs in separate environments and preserves training RNG state.

Defaults are one million requested training steps, discount `0.999`, CPU, and
SB3 MLP settings. PPO can round the budget up to a complete rollout. Use
`--timesteps`, `--seed`, `--gamma`, and `--device` to adjust these settings.

Every 25,000 steps, and after the final optimizer update, a frozen policy is
evaluated deterministically from graph-distance initialization.
`--validation-every` changes this cadence; zero requests final validation only.
`--eval-steps` sets the recurrence search budget (default 50,000).

Training retains the nondominated discovered average/peak metric points within
`A <= A* + W`, where `A*` is the best discovered continuous average AoI.
`--average-window` sets `W` (default `1`), a nonnegative finite value parsed as an
exact rational. The boundary is inclusive: `--average-window 0.5` allows half an
age unit; `--average-window 0` keeps only the best average, preferring the lowest
peak among equal averages.

A point dominates another if its average and peak are both no larger and at
least one is strictly smaller. Additional retained schedules therefore offer
lower peaks at higher averages. Identical metric points keep the first schedule.
Comparisons use rational averages and integer peaks. As `A*` improves, dominated
and out-of-window schedules are discarded. The window affects reporting only;
the reward continues to depend only on average age.

A finite run need not discover a recurrent schedule. If none is verified, the
output contains `graph`, `training`, `"schedules": []`, and
`"status": "evaluation_limit"`. This does not establish divergence.

## Optional physical-progress masking

All `N*N` actions are represented: pair `(i, j)` has index `i*N + j`.
By default, PPO and DQN can select any pair, including redundant relays.
`--progress-mask` applies the following rule to either algorithm, using the exact
pre-slot integer ages `A[monitor, process]`:

- Every fresh-source action `(j, j)` is legal, including consecutive updates.
- A relay `(i, j)` is legal exactly when some neighbor `m != j` satisfies
  `A[i,j] < A[m,j]`.

Both ages increment by one in the physical transition, so the strict inequality
means at least one receiver improves relative to passive aging. Equal-age copies
do not count. Sources can interleave freely, with no required dissemination order,
deadline, or fairness rule. The mask restricts the learner's choices; the physical
environment accepts every transmission.

PPO uses the stored mask for sampling, action likelihoods, policy updates,
entropy, and deterministic prediction. DQN explores uniformly among legal actions
and chooses greedy actions from that set. Its standard DQN target maximizes
target-network Q over legal next-state actions; it does not use Double DQN.
Masks are computed from integer physical states and stored with the corresponding
observations and replay successors. They are separate from neural-network features.

In controlled path-graph experiments, observation scaling improved PPO on P7,
and masking accelerated recurrence discovery on P7 and P8. Unmasked PPO sometimes
found lower-average schedules, including schedules that also satisfied the mask.
These observations do not establish superiority on other topologies or guarantee
that masking preserves an optimal schedule. Its benefit for DQN has not been
established experimentally.

## Periodic schedules and exact metrics

```bash
python -m aoi evaluate ppo.json
python -m aoi evaluate ppo.json --schedule-index 1
```

A successful learned result has this schema (a two-node example):

```json
{
  "graph": {"num_nodes": 2, "edges": [[0, 1]]},
  "training": {"algorithm": "ppo", "reward_divisor": 2, "observation_divisor": 2, "progress_mask": false},
  "schedules": [{
    "method": "ppo",
    "cycle_actions": [[1, 1], [0, 0]],
    "period": 2,
    "continuous_average_aoi": "2",
    "continuous_average_aoi_decimal": 2.0,
    "continuous_peak_aoi": 3
  }]
}
```

`schedules[0]` is the minimum-average schedule discovered in the run. Subsequent
entries are nondominated lower-peak alternatives within the average window,
ordered by exact average then peak. Each entry identifies its `method` (`ppo`,
`dqn`, or `algorithm1`). Index 0 is the default; `--schedule-index 1` requires a
second entry. Invalid indices and empty arrays are rejected.

Pairs are zero-based. For learned results, `cycle_actions` preserves the complete
action sequence of the observed policy-state cycle, in chronological order and
without rotation. This sequence can repeat forever.

`period` is the physical recurrent period found by independent fixed-word replay,
which first removes exact repeated copies of a shorter word. It can differ from
`len(cycle_actions)`. Replay checks full integer states at matching word phase and
verifies exact closure. If the cycle spans multiple word repetitions, all of them
contribute to its period.

Policy evaluation detects recurrence from the complete integer physical state,
starting at the initial boundary. Repeated floating-point observations, actions,
or rewards do not establish recurrence. Each detected word is independently
replayed using unrestricted dynamics and must agree exactly on both metrics.
This policy evaluator supports only stationary, memoryless policies.

For a verified recurrent period `P` and `M = N*(N-1)` ages:

- Continuous average AoI is `sum(all recurrent ages)/(P*M) + 1/2`, computed with
  `Fraction`. It is serialized as an authoritative exact rational string plus
  `continuous_average_aoi_decimal`, rounded directly from the fraction to six
  places. Bounds use the same `_decimal` convention. Comparisons and Pareto
  decisions use exact values.
- Continuous supremum peak AoI is `max(all recurrent ages) + 1`, an integer.

Here, recurrent ages are the boundary ages over one closed cycle. Ages rise
linearly between boundaries; the peak is the supremum just before receptions,
not a mean of peaks. Both metrics exclude the transient. Recurrence verifies the
observed trajectory, not global optimality or convergence from every initialization.

Evaluation recomputes metrics from the saved actions and graph. It prints a
self-contained document with the graph and one selected schedule; `--output`
also saves it to a new file:

```bash
python -m aoi evaluate algorithm1.json --output evaluated.json
```

The output preserves the selected entry's method and omits bounds and training
metadata. A standalone word uses method `evaluation`. For example, save
`{"cycle_actions": [[0, 0], [1, 1]]}` as `schedule.json` and supply its graph:

```bash
python -m aoi evaluate schedule.json --edges "0-1"
```

An external graph can also be supplied with `--graph`. If the input embeds a
graph, any external graph must match its labeled adjacency matrix exactly.
`--edges` and `--graph` are mutually exclusive. Standalone words accept only index 0.

`--max-repetitions` sets the fixed-word recurrence budget (default 1,000
repetitions of the shortest repeating word). Exhaustion produces `graph`, an
empty `schedules` array, and `"status": "evaluation_limit"`; it does not classify
the word as bounded or unbounded. A result with no schedule cannot be replayed.
Commands print JSON to stdout and decimal-first average/bound summaries with
exact fractions to stderr. They exit with code 2 for invalid input or when no
verified schedule is produced.

## Python API

The Python APIs accept NumPy adjacency matrices. For example:

```python
from aoi import AoIEnv, evaluate_policy, evaluate_schedule, progress_mask
from aoi.algorithm1 import algorithm1_schedule
from aoi.graphs import parse_edges
from aoi.train import make_model, train

adjacency = parse_edges("0-1,1-2,2-3")
construction = algorithm1_schedule(adjacency)
result = evaluate_schedule(adjacency, construction.actions)
print(result.metrics.continuous_average_aoi)  # fractions.Fraction

selected = train(adjacency, "result.json", algorithm="dqn", progress_mask=True)
```

An `env = AoIEnv(adjacency)` instance returns raw floating-point age observations
and raw rewards. `env.model.transition(state, action)` computes the pure integer transition;
`progress_mask(env.model, env.state)` returns Boolean action support without
modifying state. Training wrappers apply numerical scaling.

`make_model(AoIEnv(adjacency), ...)` constructs an in-memory SB3 model with the
chosen algorithm, masking, and independent reward and observation divisors.
`None` selects automatic scaling; a positive value overrides the divisor.
`evaluate_policy(model, AoIEnv(adjacency), max_steps=50000)` evaluates a frozen
model in a separate environment, using that model's observation divisor and mask.

`train` accepts the same algorithm, masking, and scaling choices, plus
`algorithm_options` for SB3 numerical settings, `average_window="1"` for the exact
window, and `validation_every=25000`. It returns a `TrainingResult` whose
`schedules` holds the sorted frontier of `ScheduleResult` objects and whose
`to_dict()` gives the saved JSON schema.

"""Algorithm 1: source-containing minimum connected dominating sets and DFS."""
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

from .env import validate_adjacency


@dataclass(frozen=True)
class Algorithm1Result:
    actions: tuple[tuple[int, int], ...]
    gamma_c: int
    pseudoleaves: tuple[int, ...]

    @property
    def t_star(self):
        return len(self.actions)


def algorithm1_schedule(adjacency) -> Algorithm1Result:
    """Construct a deterministic Algorithm 1 word, preserving input node labels.

    Ties use lexicographically first MCDSs and ascending DFS neighbors. Exact
    MCDS enumeration can take exponential time; no approximation is substituted.

    ``adjacency`` must satisfy AoIEnv's graph requirements. Returns an
    Algorithm1Result with the word, minimum connected domination number gamma_c,
    and pseudoleaves L (vertices in no MCDS, not necessarily graph leaves).
    Its word length is T* = N*gamma_c + |L|; AoI metrics require separate replay.
    """
    graph = nx.from_numpy_array(validate_adjacency(adjacency))
    n = len(graph)
    first = None
    containing = [None] * n
    for size in range(1, n + 1):
        for subset in combinations(range(n), size):
            members = set(subset)
            if not nx.is_connected(graph.subgraph(subset)):
                continue
            if not all(v in members or any(u in members for u in graph[v]) for v in graph):
                continue
            if first is None:
                first = subset
            for v in subset:
                if containing[v] is None:
                    containing[v] = subset
        if first is not None:
            break
    if first is None:
        raise RuntimeError("Connected graph has no connected dominating set.")
    pseudoleaves = tuple(v for v in range(n) if containing[v] is None)
    # Connectivity lets informed vertices relay; domination reaches every node.
    # A pseudoleaf adds one transmitter to an MCDS. Domination guarantees that
    # this added source touches the backbone, keeping the augmented set connected.
    actions = []
    for source in range(n):
        backbone = set(containing[source] or first) | {source}
        visited, stack = set(), [source]
        # DFS emits each relay once after an informed predecessor. Backtracking
        # needs no transmission: holders retain the update across DFS branches.
        while stack:
            tx = stack.pop()
            if tx in visited:
                continue
            visited.add(tx)
            actions.append((tx, source))
            stack.extend(sorted((set(graph[tx]) & backbone) - visited, reverse=True))
    if len(actions) != n * len(first) + len(pseudoleaves):
        raise RuntimeError("Algorithm 1 word does not have length T*.")
    return Algorithm1Result(tuple(actions), len(first), pseudoleaves)

"""Labeled edge-list JSON and adjacency-matrix conversion."""
import json
from pathlib import Path
import re

import numpy as np

from .env import validate_adjacency


def graph_from_dict(data):
    """Validate the strict JSON graph schema without dropping or relabeling edges."""
    if not isinstance(data, dict) or set(data) != {"num_nodes", "edges"}:
        raise ValueError("Graph must be an object with exactly num_nodes and edges fields.")
    n, edges = data["num_nodes"], data["edges"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("Graph num_nodes must be an integer of at least two.")
    if not isinstance(edges, list):
        raise ValueError("Graph edges must be a list of integer vertex pairs.")
    seen = set()
    for edge in edges:
        if (not isinstance(edge, list) or len(edge) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in edge)):
            raise ValueError("Each graph edge must be a pair of integer vertex indices (not Booleans).")
        u, v = edge
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("Graph edge endpoints must be in 0 through num_nodes - 1.")
        if u == v:
            raise ValueError("Graph self-loops are not allowed.")
        pair = (min(u, v), max(u, v))
        if pair in seen:
            raise ValueError(f"Duplicate graph edge: {pair} (including reversed duplicates).")
        seen.add(pair)
    adjacency = np.zeros((n, n), dtype=bool)
    for u, v in seen:
        adjacency[u, v] = adjacency[v, u] = True
    return validate_adjacency(adjacency)


def parse_edges(value):
    """Parse only comma-separated u-v pairs, inferring N from the largest endpoint."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Edges must be a nonempty comma-separated list of u-v pairs.")
    edges = []
    for token in value.split(","):
        match = re.fullmatch(r"\s*([0-9]+)\s*-\s*([0-9]+)\s*", token)
        if match is None:
            raise ValueError("Edges must use comma-separated u-v pairs with nonnegative integer indices.")
        edges.append([int(match[1]), int(match[2])])
    return graph_from_dict({"num_nodes": 1 + max(max(e) for e in edges), "edges": edges})


def graph_to_dict(adjacency):
    """Serialize edges in sorted order while preserving node labels."""
    adjacency = validate_adjacency(adjacency)
    return {"num_nodes": len(adjacency),
            "edges": np.argwhere(np.triu(adjacency, k=1)).tolist()}


def read_json(path):
    """Read JSON, rejecting duplicate fields and nonfinite numeric constants."""
    def object_fields(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"Duplicate JSON field: {key}.")
            data[key] = value
        return data

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}.")

    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=object_fields,
                          parse_constant=invalid_constant)
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def read_graph(path):
    return graph_from_document(read_json(path))


def graph_from_document(data):
    """Read the required graph field independently of other document contents."""
    if not isinstance(data, dict) or "graph" not in data:
        raise ValueError("Unified JSON document must contain a graph field.")
    return graph_from_dict(data["graph"])


def write_json(data, path):
    """Write JSON to a new file, raising FileExistsError if the path exists."""
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as stream:
        stream.write(text)

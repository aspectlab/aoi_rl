import json

import networkx as nx
import numpy as np
import pytest

from aoi.graphs import graph_from_dict, graph_to_dict, parse_edges, read_graph


@pytest.mark.parametrize('graph', [nx.path_graph(6), nx.cycle_graph(5),
                                   nx.star_graph(4), nx.complete_graph(4)])
def test_graph_inputs_and_canonical_round_trip(graph, tmp_path):
    adjacency = nx.to_numpy_array(graph)
    edges = [[v, u] for u, v in reversed(list(graph.edges))]
    data = {'num_nodes': len(graph), 'edges': edges}
    inline = ' , '.join(f' {u} - {v} ' for u, v in edges)
    path = tmp_path/'graph.json'
    path.write_text(json.dumps({"graph": data}))
    for actual in [parse_edges(inline), graph_from_dict(data), read_graph(path)]:
        np.testing.assert_array_equal(actual, adjacency)
        canonical = graph_to_dict(actual)
        assert canonical == {'num_nodes': len(graph), 'edges': sorted(map(sorted, edges))}
        np.testing.assert_array_equal(graph_from_dict(canonical), adjacency)


def test_labels_are_preserved_not_isomorphism_canonicalized():
    first, second = parse_edges('0-1,1-2'), parse_edges('0-2,2-1')
    assert nx.is_isomorphic(nx.from_numpy_array(first), nx.from_numpy_array(second))
    assert graph_to_dict(first) != graph_to_dict(second)
    assert graph_to_dict(second)['edges'] == [[0, 2], [1, 2]]


@pytest.mark.parametrize('edges', ['', ' ', '0', '0-1,', ',0-1', '0-1,,1-2',
    '0-1-2', '-1-0', '0--1', '0-1.0', '0-1e0', '0-True', '0:1', '[0,1]',
    '0–1', '0-+1', '0-1;1-2', '0-1 1-2', 'a-b', '0-１',
    '0-0', '0-1,0-1', '0-1,1-0', '0-1,2-3', '1-2', '0-2'])
def test_invalid_inline_edges(edges):
    with pytest.raises(ValueError):
        parse_edges(edges)


@pytest.mark.parametrize('data', [None, [], 2, 'graph', {}, {'edges': [[0, 1]]},
    {'num_nodes': 2}, {'num_nodes': 2, 'edges': [[0, 1]], 'extra': 1},
    *[{'num_nodes': n, 'edges': [[0, 1]]} for n in [True, False, 2.0, '2', 0, -1, 1, None]],
    *[{'num_nodes': 2, 'edges': e} for e in [None, {}, '0-1', [0, 1], [[0]], [[0, 1, 2]],
        [[0, True]], [[False, 1]], [[0, 1.0]], [[0, '1']], [[None, 1]], [[-1, 0]],
        [[0, 2]], [[0, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], []]],
    {'num_nodes': 3, 'edges': [[0, 1]]}, {'num_nodes': 4, 'edges': [[0, 1], [2, 3]]},
    {'num_nodes': 3, 'edges': [[0, 2]]}])
def test_invalid_graph_structures(data):
    with pytest.raises(ValueError):
        graph_from_dict(data)


@pytest.mark.parametrize('text', ['', '{', 'null', '[]',
    '{"num_nodes":2,"edges":[[0,1]],}',
    '{"num_nodes":2,"num_nodes":3,"edges":[[0,1]]}',
    '{"num_nodes":2,"edges":[[0,NaN]]}'])
def test_invalid_json_graph_files(tmp_path, text):
    path = tmp_path/'graph.json'
    path.write_text(text)
    with pytest.raises(ValueError):
        read_graph(path)


def test_serialization_uses_existing_matrix_validation():
    for adjacency in [[[0, 1], [0, 0]], [[0, 2], [2, 0]], [[0, 0], [0, 0]]]:
        with pytest.raises(ValueError):
            graph_to_dict(adjacency)

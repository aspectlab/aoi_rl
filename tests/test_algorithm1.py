import networkx as nx
from aoi.algorithm1 import algorithm1_schedule
from aoi.schedules import evaluate_schedule


def test_dfs_branches_and_augmentation():
    graph = nx.Graph([(0, 1), (1, 4), (0, 2), (4, 3), (2, 5)])
    a = nx.to_numpy_array(graph, nodelist=range(6))
    result = algorithm1_schedule(a)
    assert [i for i, j in result.actions if j == 0] == [0, 1, 4, 2]
    assert result.pseudoleaves == (3, 5)
    assert result.t_star == 6 * result.gamma_c + 2
    for j in range(6):
        informed = {j}
        block = [i for i, p in result.actions if p == j]
        assert block[0] == j
        assert len(block) == result.gamma_c + (j in result.pseudoleaves)
        for i in block:
            assert i in informed
            informed.update(graph[i])
        assert informed == set(graph)
    assert evaluate_schedule(a, result.actions).metrics.period == result.t_star


def test_multiple_minimum_backbones_use_source_containing_choice():
    a = nx.to_numpy_array(nx.cycle_graph(4))
    result = algorithm1_schedule(a)
    assert result.gamma_c == 2
    assert result.pseudoleaves == ()
    assert result.actions == ((0, 0), (1, 0), (1, 1), (0, 1),
                              (2, 2), (1, 2), (3, 3), (0, 3))

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


def test_exact_average_bounds_and_decimal_serialization(tmp_path, capsys):
    import json
    from fractions import Fraction
    from aoi.__main__ import main
    from aoi.algorithm1 import average_age_bounds, mean_shortest_path_distance

    for n, distance, lower, upper, average, peak in [
        (3, '4/3', '23/6', '25/6', '23/6', 7),
        (6, '7/3', '46/3', '52/3', '232/15', 31),
    ]:
        adjacency = nx.to_numpy_array(nx.path_graph(n))
        construction = algorithm1_schedule(adjacency)
        assert construction.gamma_c == n - 2
        assert construction.pseudoleaves == (0, n - 1)
        assert construction.t_star == n * (n - 2) + 2
        assert mean_shortest_path_distance(adjacency) == Fraction(distance)
        bounds = average_age_bounds(adjacency, construction)
        assert list(bounds.values()) == [Fraction(lower), Fraction(upper)]
        metrics = evaluate_schedule(adjacency, construction.actions).metrics
        assert Fraction(lower) <= metrics.continuous_average_aoi <= Fraction(upper)
        assert metrics.continuous_average_aoi == Fraction(average)
        assert metrics.continuous_peak_aoi == peak
        output = tmp_path / f'p{n}.json'
        assert main(['algorithm1', '--edges', ','.join(f'{i}-{i+1}' for i in range(n-1)),
                     '--output', str(output)]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data == json.loads(output.read_text())
        for key, value in bounds.items():
            assert data['bounds'][key] == str(value)
            assert data['bounds'][key + '_decimal'] == float(round(value, 6))
        entry = data['schedules'][0]
        assert entry['continuous_average_aoi'] == average
        assert entry['continuous_average_aoi_decimal'] == float(round(Fraction(average), 6))
        assert entry['cycle_actions'] == [list(a) for a in construction.actions]


def test_bounds_use_all_minimum_sets_not_graph_leaves():
    from fractions import Fraction
    from aoi.algorithm1 import average_age_bounds, mean_shortest_path_distance
    # C4 has no pseudoleaves: every vertex belongs to a minimum backbone.
    adjacency = nx.to_numpy_array(nx.cycle_graph(4))
    result = algorithm1_schedule(adjacency)
    assert result.gamma_c == 2 and result.pseudoleaves == ()
    assert mean_shortest_path_distance(adjacency) == Fraction(4, 3)
    assert list(average_age_bounds(adjacency, result).values()) == [Fraction(16, 3), Fraction(6)]
    # A triangle with two attached triangles has pseudoleaves of degree two.
    graph = nx.Graph([(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    adjacency = nx.to_numpy_array(graph, nodelist=range(5))
    result = algorithm1_schedule(adjacency)
    assert result.gamma_c == 1 and result.pseudoleaves == (1, 2, 3, 4)
    assert mean_shortest_path_distance(adjacency) == Fraction(7, 5)
    assert list(average_age_bounds(adjacency, result).values()) == [Fraction(59, 10), Fraction(63, 10)]

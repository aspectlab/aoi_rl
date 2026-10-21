import json
import subprocess
import sys

import pytest
from aoi.__main__ import main
from aoi.schedules import schedule_entry


GRAPH = {"num_nodes": 2, "edges": [[0, 1]]}


@pytest.mark.parametrize('source', ['edges', 'graph'])
def test_algorithm_schedule_round_trip(tmp_path, capsys, source):
    graph = tmp_path/'graph.json'
    graph.write_text(json.dumps({"graph": {"num_nodes": 3, "edges": [[0, 1], [1, 2]]}}))
    output = tmp_path/'schedule.json'
    args = ['--edges', '2-1,1-0'] if source == 'edges' else ['--graph', str(graph)]
    assert main(['algorithm1', *args, '--output', str(output)]) == 0
    expected = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == expected
    assert expected['graph'] == {'num_nodes': 3, 'edges': [[0, 1], [1, 2]]}
    assert json.dumps(expected).count('"graph"') == 1
    assert 'transient_actions' not in expected
    assert main(['evaluate', str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == {k: v for k, v in expected.items() if k != "bounds"}
    with pytest.raises(SystemExit):
        main(['algorithm1', '--graph', str(graph), '--output', str(output)])


def test_evaluation_limit_json(tmp_path, capsys):
    graph, schedule = tmp_path/'graph.json', tmp_path/'word.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    schedule.write_text(json.dumps({'cycle_actions': [[0, 0]]}))
    assert main(['evaluate', str(schedule), '--graph', str(graph), '--max-repetitions', '2']) == 2
    assert json.loads(capsys.readouterr().out) == {'graph': GRAPH, 'schedules': [], 'status': 'evaluation_limit'}


def test_learned_json_can_be_evaluated_without_its_transient(tmp_path, capsys):
    graph, schedule = tmp_path/'graph.json', tmp_path/'learned.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    from aoi.env import AoIEnv
    from aoi.policy_eval import evaluate_policy
    class Oldest:
        def predict(self, obs, deterministic=True):
            return (0 if obs[1] >= obs[0] else 3), None
    result = evaluate_policy(Oldest(), AoIEnv([[0, 1], [1, 0]]), max_steps=10)
    schedule.write_text(json.dumps(result.to_dict()))
    assert main(['evaluate', str(schedule), '--graph', str(graph)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"graph": GRAPH, "schedules": [schedule_entry(result.schedule, "evaluation")]}
    assert 'transient_actions' not in output


def test_module_entrypoint():
    result = subprocess.run([sys.executable, '-m', 'aoi', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(command in result.stdout for command in ['algorithm1', 'evaluate', 'train'])
    assert 'policy' not in result.stdout


@pytest.mark.parametrize('source', ['edges', 'graph'])
@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('masked', [False, True])
def test_training_command_writes_one_file(tmp_path, capsys, algorithm, masked, source):
    graph, output = tmp_path/'graph.json', tmp_path/'result.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    graph_args = ['--edges', '1-0'] if source == 'edges' else ['--graph', str(graph)]
    args = ['train', *graph_args, '--algorithm', algorithm, '--timesteps', '8',
            '--output', str(output), '--eval-steps', '20', '--validation-every', '0']
    if masked:
        args.append('--progress-mask')
    code = main(args)
    result = json.loads(capsys.readouterr().out)
    assert result['graph'] == GRAPH
    assert json.dumps(result).count('"graph"') == 1
    assert code == (0 if result['schedules'] else 2)
    if code == 0:
        assert main(['evaluate', str(output)]) == 0
        replay = json.loads(capsys.readouterr().out)
        assert replay == {'graph': GRAPH, 'schedules': [result['schedules'][0]]}
    else:
        assert result == {'graph': GRAPH, 'schedules': [], 'status': 'evaluation_limit',
                          'training': {'algorithm': algorithm, 'reward_divisor': 2,
                                       'observation_divisor': 2, 'progress_mask': masked}}
    assert json.loads(output.read_text()) == result
    assert set(tmp_path.iterdir()) == {graph, output}


def test_empty_graph_file_reports_input_error(tmp_path):
    graph = tmp_path/'empty.json'
    graph.touch()
    result = subprocess.run([sys.executable, '-m', 'aoi', 'algorithm1', '--graph', str(graph)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Expecting value' in result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stdout == ''


@pytest.mark.parametrize('option, value', [
    ('--reward-divisor', '0'), ('--reward-divisor', '-1'),
    ('--reward-divisor', 'nan'), ('--reward-divisor', 'inf'),
    ('--reward-divisor', 'oops'), ('--observation-divisor', '0'),
    ('--observation-divisor', '-1'), ('--observation-divisor', 'nan'),
    ('--observation-divisor', 'inf'), ('--observation-divisor', 'oops'), ('--average-window', '-1'),
    ('--average-window', 'nan'), ('--average-window', 'inf'),
    ('--average-window', 'oops')])
def test_invalid_training_numbers(tmp_path, capsys, option, value):
    graph, output = tmp_path/'graph.json', tmp_path/'result.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    with pytest.raises(SystemExit) as exc:
        main(['train', '--graph', str(graph), '--output', str(output), option, value])
    assert exc.value.code == 2
    assert 'must be' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize('index', ['-1', '1', 'no'])
def test_invalid_schedule_index(tmp_path, capsys, index):
    graph, schedule = tmp_path/'graph.json', tmp_path/'schedule.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    schedule.write_text(json.dumps({'graph': GRAPH, 'schedules': [{'cycle_actions': [[0, 0], [1, 1]]}]}))
    with pytest.raises(SystemExit) as exc:
        main(['evaluate', str(schedule), '--graph', str(graph), '--schedule-index', index])
    assert exc.value.code == 2
    assert 'schedule-index' in capsys.readouterr().err


def test_training_frontier_json_independent_replay(tmp_path, monkeypatch, capsys):
    import importlib
    from aoi.policy_eval import PolicyResult
    from aoi.schedules import evaluate_schedule
    from aoi.train import train
    adjacency = [[0, 1], [1, 0]]
    # Real periodic words: better average / better peak / dominated.
    words = [[0, 0, 0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 1, 1, 1],
             [0, 0, 0, 0, 0, 1, 1, 1]]
    verified = [evaluate_schedule(adjacency, [(i, i) for i in word]) for word in words]
    discoveries = iter([verified[1], verified[0], verified[2]])
    # Feed deterministic verified discoveries at real training validation points.
    monkeypatch.setattr(importlib.import_module('aoi.train'), 'evaluate_policy',
                        lambda *args, **kwargs: PolicyResult(next(discoveries)))
    graph, output = tmp_path/'graph.json', tmp_path/'result.json'
    graph.write_text(json.dumps({"graph": GRAPH}))
    result = train(adjacency, output, timesteps=8, validation_every=4,
                   algorithm_options={'n_steps': 8, 'batch_size': 4, 'n_epochs': 1})
    assert result.schedules == verified[:2]
    data = json.loads(output.read_text())
    assert 'transient' not in output.read_text()
    assert len(data['schedules']) == 2
    for index, expected in enumerate(verified[:2]):
        assert main(['evaluate', str(output), '--schedule-index', str(index)]) == 0
        assert json.loads(capsys.readouterr().out) == {"graph": GRAPH, "schedules": [schedule_entry(expected, "ppo")]}
    assert main(['evaluate', str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == {"graph": GRAPH, "schedules": [schedule_entry(verified[0], "ppo")]}
    assert set(tmp_path.iterdir()) == {graph, output}


@pytest.mark.parametrize('command', ['train', 'algorithm1'])
@pytest.mark.parametrize('args', [[], ['--edges', '0-1', '--graph', 'graph.json'],
    ['old.npy'], ['--edges', '0-1,1-0'], ['--edges', '1-2'], ['--edges', '0:1']])
def test_graph_input_cli_errors(command, args, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, *args])
    assert exc.value.code == 2
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize('source', ['edges', 'graph'])
@pytest.mark.parametrize('embedded', [False, True])
def test_external_graph_and_independent_metrics(tmp_path, capsys, source, embedded):
    schedule, graph = tmp_path/'schedule.json', tmp_path/'graph.json'
    graph.write_text(json.dumps({'graph': {'num_nodes': 2, 'edges': [[1, 0]]}}))
    data = {'cycle_actions': [[0, 0], [1, 1]], 'period': 999,
            'continuous_average_aoi': '999', 'continuous_peak_aoi': 999}
    if embedded:
        data = {'graph': GRAPH, 'schedules': [data]}
    schedule.write_text(json.dumps(data))
    args = ['--edges', '1-0'] if source == 'edges' else ['--graph', str(graph)]
    output = tmp_path/'evaluated.json'
    assert main(['evaluate', str(schedule), *args, '--output', str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {'graph': GRAPH, 'schedules': [{'method': 'evaluation',
                      'cycle_actions': [[0, 0], [1, 1]], 'period': 2,
                      'continuous_average_aoi': '2', 'continuous_average_aoi_decimal': 2.0, 'continuous_peak_aoi': 3}]}
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize('external', ['0-2,2-1', '0-1', '0-1,1-2,0-2'])
def test_mismatched_graphs_rejected(tmp_path, capsys, external):
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps({'graph': {'num_nodes': 3, 'edges': [[0, 1], [1, 2]]},
                                    'cycle_actions': [[0, 0]]}))
    with pytest.raises(SystemExit) as exc:
        main(['evaluate', str(schedule), '--edges', external])
    assert exc.value.code == 2
    assert 'must match exactly' in capsys.readouterr().err


@pytest.mark.parametrize('data, args, message', [
    ({'cycle_actions': [[0, 0]]}, [], 'no embedded graph'),
    ({'graph': GRAPH, 'schedules': [], 'status': 'evaluation_limit'}, [], 'no verified schedule'),
    ({'graph': GRAPH, 'schedules': [{'cycle_actions': [[0, 0]]}]}, ['--schedule-index', '1'], 'out of range'),
    ({'graph': GRAPH, 'schedules': {}}, [], 'must be a list'),
    ({'graph': GRAPH, 'schedules': [None]}, [], 'cycle_actions'),
    ({'graph': GRAPH, 'best_average': {}}, [], 'no verified schedule'),
    ({'schedules': []}, ['--edges', '0-1'], 'graph field'),
    ({'graph': None, 'cycle_actions': []}, ['--edges', '0-1'], 'Graph must'),
    ({'cycle_actions': []}, ['--edges', '0-1', '--graph', 'unused.json'], 'not allowed'),
    ({'graph': GRAPH, 'schedules': [{'cycle_actions': [[True, 0]]}]}, [], 'integer'),
])
def test_evaluation_input_errors(tmp_path, capsys, data, args, message):
    path = tmp_path/'schedule.json'
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as exc:
        main(['evaluate', str(path), *args])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_frontier_metrics_are_not_trusted(tmp_path, capsys):
    path = tmp_path/'frontier.json'
    entry = {'cycle_actions': [[0, 0], [1, 1]], 'period': -1,
             'continuous_average_aoi': 'wrong', 'continuous_peak_aoi': 100}
    path.write_text(json.dumps({'graph': GRAPH, 'status': 'cycle',
                               'schedules': [entry, entry]}))
    for args in [[], ['--schedule-index', '1']]:
        assert main(['evaluate', str(path), *args]) == 0
        result = json.loads(capsys.readouterr().out)['schedules'][0]
        assert result['period'] == 2
        assert result['continuous_average_aoi'] == '2'
        assert result['continuous_peak_aoi'] == 3


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
def test_training_failure_embeds_graph(tmp_path, capsys, algorithm):
    output = tmp_path/'failure.json'
    assert main(['train', '--edges', '0-1', '--algorithm', algorithm,
                 '--timesteps', '8', '--eval-steps', '1', '--validation-every', '0',
                 '--output', str(output)]) == 2
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text()) == {
        'graph': GRAPH, 'schedules': [], 'status': 'evaluation_limit',
        'training': {'algorithm': algorithm, 'reward_divisor': 2,
                     'observation_divisor': 2, 'progress_mask': False}}
    with pytest.raises(SystemExit):
        main(['evaluate', str(output)])
    assert 'no verified schedule' in capsys.readouterr().err


@pytest.mark.parametrize('source', ['graph', 'algorithm1', 'ppo', 'dqn', 'evaluation'])
def test_graph_reuse_only_reads_graph(tmp_path, monkeypatch, capsys, source):
    import importlib
    from aoi.policy_eval import PolicyResult
    from aoi.schedules import evaluate_schedule
    from aoi.graphs import read_graph
    import numpy as np

    adjacency = [[0, 1], [1, 0]]
    verified = evaluate_schedule(adjacency, [(0, 0), (1, 1)])
    data = {'graph': {'num_nodes': 2, 'edges': [[1, 0]]}}
    if source != 'graph':
        data.update(bounds={'old_bound': '999'}, schedules=[schedule_entry(verified, source)],
                    training={'algorithm': 'old', 'reward_divisor': 999, 'observation_divisor': 999,
                              'progress_mask': True})
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    np.testing.assert_array_equal(read_graph(path), adjacency)
    monkeypatch.setattr(importlib.import_module('aoi.train'), 'evaluate_policy',
                        lambda *a, **kw: PolicyResult(verified))
    for seed in (0, 1):
        output = tmp_path / f'run{seed}.json'
        assert main(['train', '--graph', str(path), '--algorithm', 'dqn', '--seed', str(seed),
                     '--timesteps', '8', '--validation-every', '0', '--output', str(output)]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result == {'graph': GRAPH, 'schedules': [schedule_entry(verified, 'dqn')],
                          'training': {'algorithm': 'dqn', 'reward_divisor': 2,
                                       'observation_divisor': 2, 'progress_mask': False}}
        assert json.loads(output.read_text()) == result
    assert path.read_bytes() == original


@pytest.mark.parametrize('command', ['algorithm1', 'train', 'evaluate'])
@pytest.mark.parametrize('collision', [False, True])
def test_no_output_overwrite_even_on_failure(tmp_path, capsys, command, collision):
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'graph': GRAPH, 'schedules': [{'cycle_actions': [[0, 0]]}]}))
    output = source if collision else tmp_path / 'existing.json'
    if not collision:
        output.write_text('prior results\n')
    original = output.read_bytes()
    args = [str(source), '--max-repetitions', '1'] if command == 'evaluate' else ['--graph', str(source)]
    with pytest.raises(SystemExit) as exc:
        main([command, *args, '--output', str(output)])
    assert exc.value.code == 2
    assert 'already exists' in capsys.readouterr().err
    assert output.read_bytes() == original


@pytest.mark.parametrize('data', [{}, GRAPH, {'schedules': []}, {'graph': None}])
def test_unified_graph_is_required(tmp_path, data):
    from aoi.graphs import read_graph
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='[Gg]raph'):
        read_graph(path)


def test_decimal_first_terminal_summary(tmp_path, capsys):
    output = tmp_path / 'p3.json'
    assert main(['algorithm1', '--edges', '0-1,1-2', '--output', str(output)]) == 0
    captured = capsys.readouterr()
    assert 'Average AoI: 3.833333 (23/6)' in captured.err
    assert 'average_lower_tstar_periodic: 3.833333 (23/6)' in captured.err
    assert 'average_upper_algorithm1: 4.166667 (25/6)' in captured.err
    assert main(['evaluate', str(output)]) == 0
    captured = capsys.readouterr()
    assert 'Average AoI: 3.833333 (23/6)' in captured.err
    entry = json.loads(captured.out)['schedules'][0]
    assert entry['continuous_average_aoi_decimal'] == 3.833333


def test_default_ppo_and_removed_flag(tmp_path, capsys):
    output = tmp_path / 'default.json'
    main(['train', '--edges', '0-1', '--timesteps', '8', '--eval-steps', '10',
          '--validation-every', '0', '--output', str(output)])
    data = json.loads(capsys.readouterr().out)
    assert data['training'] == {'algorithm': 'ppo', 'reward_divisor': 2,
                                'observation_divisor': 2, 'progress_mask': False}
    with pytest.raises(SystemExit) as exc:
        main(['train', '--edges', '0-1', '--connected-block'])
    assert exc.value.code == 2
    assert 'unrecognized arguments: --connected-block' in capsys.readouterr().err


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('arguments,reward,observation', [
    (['--reward-divisor', '1'], 1, 2),
    (['--observation-divisor', '1'], 2, 1),
    (['--reward-divisor', '1', '--observation-divisor', '56'], 1, 56)])
def test_cli_independent_divisor_overrides(tmp_path, capsys, algorithm, arguments, reward, observation):
    output = tmp_path / 'scaled.json'
    main(['train', '--edges', '0-1', '--algorithm', algorithm, '--timesteps', '8',
          '--eval-steps', '10', '--validation-every', '0', '--output', str(output), *arguments])
    assert json.loads(capsys.readouterr().out)['training'] == {
        'algorithm': algorithm, 'reward_divisor': reward,
        'observation_divisor': observation, 'progress_mask': False}

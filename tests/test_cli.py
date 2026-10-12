import json
import subprocess
import sys

import numpy as np
import pytest
from aoi.__main__ import main


def test_algorithm_schedule_round_trip(tmp_path, capsys):
    graph = tmp_path/'graph.npy'
    np.save(graph, [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    output = tmp_path/'schedule.json'
    assert main(['algorithm1', str(graph), '--output', str(output)]) == 0
    expected = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == expected
    assert 'transient_actions' not in expected
    assert main(['evaluate', str(graph), str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    with pytest.raises(SystemExit):
        main(['algorithm1', str(graph), '--output', str(output)])


def test_evaluation_limit_json(tmp_path, capsys):
    graph, schedule = tmp_path/'graph.npy', tmp_path/'word.json'
    np.save(graph, [[0, 1], [1, 0]])
    schedule.write_text(json.dumps({'cycle_actions': [[0, 0]]}))
    assert main(['evaluate', str(graph), str(schedule), '--max-repetitions', '2']) == 2
    assert json.loads(capsys.readouterr().out) == {'status': 'evaluation_limit'}


def test_learned_json_can_be_evaluated_without_its_transient(tmp_path, capsys):
    graph, schedule = tmp_path/'graph.npy', tmp_path/'learned.json'
    np.save(graph, [[0, 1], [1, 0]])
    from aoi.env import AoIEnv
    from aoi.policy_eval import evaluate_policy
    class Oldest:
        def predict(self, obs, deterministic=True):
            return (0 if obs[1] >= obs[0] else 3), None
    result = evaluate_policy(Oldest(), AoIEnv([[0, 1], [1, 0]]), max_steps=10)
    schedule.write_text(json.dumps(result.to_dict()))
    assert main(['evaluate', str(graph), str(schedule)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == result.schedule.to_dict()
    assert 'transient_actions' not in output


def test_module_entrypoint():
    result = subprocess.run([sys.executable, '-m', 'aoi', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert all(command in result.stdout for command in ['algorithm1', 'evaluate', 'train'])
    assert 'policy' not in result.stdout


@pytest.mark.parametrize('algorithm', ['ppo', 'dqn'])
@pytest.mark.parametrize('restricted', [False, True])
def test_training_command_writes_one_file(tmp_path, capsys, algorithm, restricted):
    graph, output = tmp_path/'graph.npy', tmp_path/'result.json'
    np.save(graph, [[0, 1], [1, 0]])
    args = ['train', str(graph), '--algorithm', algorithm, '--timesteps', '8',
            '--output', str(output), '--eval-steps', '20', '--validation-every', '0']
    if restricted:
        args.append('--connected-block')
    code = main(args)
    result = json.loads(capsys.readouterr().out)
    assert code == (0 if result['status'] == 'cycle' else 2)
    assert json.loads(output.read_text()) == result
    assert set(tmp_path.iterdir()) == {graph, output}


def test_empty_graph_file_reports_input_error(tmp_path):
    graph = tmp_path/'empty.npy'
    graph.touch()
    result = subprocess.run([sys.executable, '-m', 'aoi', 'algorithm1', str(graph)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Graph file is empty' in result.stderr
    assert str(graph) in result.stderr
    assert '.npy adjacency matrix' in result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stdout == ''

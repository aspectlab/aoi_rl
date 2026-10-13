"""Graph edges to self-contained periodic schedules and exact AoI metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .graphs import graph_from_dict, graph_to_dict, parse_edges, read_graph, read_json
from .schedules import EvaluationLimit, evaluate_schedule


def read_schedule(path, schedule_index=0):
    return select_schedule(read_json(path), schedule_index)


def select_schedule(data, schedule_index=0):
    if (isinstance(schedule_index, bool) or not isinstance(schedule_index, int)
            or schedule_index < 0):
        raise ValueError("schedule-index must be a nonnegative integer.")
    if isinstance(data, dict) and data.get("status") == "evaluation_limit":
        raise ValueError("Result has status evaluation_limit; no verified schedule is available.")
    if isinstance(data, dict) and "best_average" in data:
        if "cycle_actions" in data:
            raise ValueError("Schedule JSON cannot contain both best_average and cycle_actions.")
        alternatives = data.get("alternatives", [])
        if not isinstance(alternatives, list):
            raise ValueError("Schedule alternatives must be a list.")
        schedules = [data["best_average"], *alternatives]
        if schedule_index >= len(schedules):
            raise ValueError("schedule-index is out of range.")
        data = schedules[schedule_index]
    elif schedule_index != 0:
        raise ValueError("schedule-index is out of range.")
    if not isinstance(data, dict) or not isinstance(data.get("cycle_actions"), list):
        raise ValueError("Schedule JSON must contain a cycle_actions list.")
    return data["cycle_actions"]


def emit(data, path=None):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if path is not None:
        with Path(path).open("x") as stream:
            stream.write(text)
    print(text, end="")


def add_graph_arguments(command, *, required):
    sources = command.add_mutually_exclusive_group(required=required)
    sources.add_argument("--edges", help='Comma-separated undirected edges, e.g. "0-1,1-2".')
    sources.add_argument("--graph", type=Path, help="JSON graph with num_nodes and edges.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Age of Information scheduling")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("algorithm1", "Construct the MCDS/DFS schedule."),
                            ("evaluate", "Evaluate an explicit periodic word.")):
        command = commands.add_parser(name, help=help_text)
        add_graph_arguments(command, required=name != "evaluate")
        if name == "evaluate":
            command.add_argument("schedule", type=Path)
            command.add_argument("--schedule-index", type=int, default=0,
                                 help="0 selects best_average; 1, 2, ... select alternatives.")
        command.add_argument("--output", type=Path)
        command.add_argument("--max-repetitions", type=int, default=1000)
    command = commands.add_parser("train", help="Train PPO or DQN and save verified schedule tradeoffs.")
    add_graph_arguments(command, required=True)
    command.add_argument("--algorithm", choices=("ppo", "dqn"), default="ppo")
    command.add_argument("--connected-block", action="store_true")
    command.add_argument("--output", type=Path, default=Path("result.json"))
    command.add_argument("--timesteps", type=int, default=1_000_000)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--gamma", type=float, default=0.999)
    command.add_argument("--validation-every", type=int, default=25_000,
                         help="Validation cadence; 0 means final validation only.")
    command.add_argument("--reward-divisor", help="Positive training divisor; default N*diameter(G).")
    command.add_argument("--average-window", default="1",
                         help="Nonnegative exact average-AoI window (default 1).")
    command.add_argument("--eval-steps", type=int, default=50_000)
    command.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        adjacency = (parse_edges(args.edges) if args.edges is not None else
                     read_graph(args.graph) if args.graph is not None else None)
        if args.command == "evaluate":
            data = read_json(args.schedule)
            if isinstance(data, dict) and "graph" in data:
                embedded = graph_from_dict(data["graph"])
                if adjacency is not None and not np.array_equal(adjacency, embedded):
                    raise ValueError("Embedded and external graphs must match exactly, including vertex labels.")
                adjacency = embedded
            if adjacency is None:
                raise ValueError("Schedule has no embedded graph; supply --edges or --graph.")
            actions = select_schedule(data, args.schedule_index)
        graph = graph_to_dict(adjacency)
        if args.command == "train":
            from .train import train
            result = train(adjacency, args.output, algorithm=args.algorithm,
                connected_block=args.connected_block, timesteps=args.timesteps,
                seed=args.seed, gamma=args.gamma, validation_every=args.validation_every,
                max_steps=args.eval_steps, device=args.device,
                reward_divisor=args.reward_divisor, average_window=args.average_window)
            emit(result.to_dict())
            return 0 if result.schedules else 2
        if args.command == "algorithm1":
            from .algorithm1 import algorithm1_schedule
            actions = algorithm1_schedule(adjacency).actions
        try:
            result = evaluate_schedule(adjacency, actions, max_repetitions=args.max_repetitions)
        except EvaluationLimit:
            emit({"graph": graph, "status": "evaluation_limit"}, args.output)
            return 2
        emit({"graph": graph, **result.to_dict()}, args.output)
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

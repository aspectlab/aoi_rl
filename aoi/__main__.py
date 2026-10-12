"""Adjacency matrix to periodic transmission schedule and exact AoI metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from .env import validate_adjacency
from .schedules import EvaluationLimit, evaluate_schedule


def read_graph(path):
    try:
        adjacency = np.load(path, allow_pickle=False)
    except EOFError as exc:
        raise ValueError(f"Graph file is empty: {path}. Supply a .npy adjacency matrix.") from exc
    return validate_adjacency(adjacency)


def read_schedule(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("cycle_actions"), list):
        raise ValueError("Schedule JSON must contain a cycle_actions list.")
    return data["cycle_actions"]


def emit(data, path=None):
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if path is not None:
        with Path(path).open("x") as stream:
            stream.write(text)
    print(text, end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Age of Information scheduling")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("algorithm1", "Construct the MCDS/DFS schedule."),
                            ("evaluate", "Evaluate an explicit periodic word.")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("graph", type=Path)
        if name == "evaluate":
            command.add_argument("schedule", type=Path)
        command.add_argument("--output", type=Path)
        command.add_argument("--max-repetitions", type=int, default=1000)
    command = commands.add_parser("train", help="Train PPO or DQN and save the best verified schedule.")
    command.add_argument("graph", type=Path)
    command.add_argument("--algorithm", choices=("ppo", "dqn"), default="ppo")
    command.add_argument("--connected-block", action="store_true")
    command.add_argument("--output", type=Path, default=Path("result.json"))
    command.add_argument("--timesteps", type=int, default=1_000_000)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--gamma", type=float, default=0.999)
    command.add_argument("--validation-every", type=int, default=100_000,
                         help="Validation cadence; 0 means final validation only.")
    command.add_argument("--eval-steps", type=int, default=50_000)
    command.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        adjacency = read_graph(args.graph)
        if args.command == "train":
            from .train import train
            result = train(adjacency, args.output, algorithm=args.algorithm,
                connected_block=args.connected_block, timesteps=args.timesteps,
                seed=args.seed, gamma=args.gamma, validation_every=args.validation_every,
                max_steps=args.eval_steps, device=args.device)
            emit(result.to_dict())
            return 0 if result.schedule is not None else 2
        if args.command == "algorithm1":
            from .algorithm1 import algorithm1_schedule
            actions = algorithm1_schedule(adjacency).actions
        else:
            actions = read_schedule(args.schedule)
        try:
            result = evaluate_schedule(adjacency, actions, max_repetitions=args.max_repetitions)
        except EvaluationLimit:
            emit({"status": "evaluation_limit"}, args.output)
            return 2
        emit(result.to_dict(), args.output)
        return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

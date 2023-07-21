import json

import toml
import yaml

import wandb
from colingo.zoo.signaling import ConfigWithMLPRNN, train_with_mlp_rnn


def sweep() -> None:
    wandb.init()
    config = dict(wandb.config)
    train_with_mlp_rnn(ConfigWithMLPRNN(**config))


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to config file", default=None)
    parser.add_argument("--id", type=str, help="Sweep ID", default=None)
    parser.add_argument(
        "--wandb_project", "-p", type=str, help="Wandb project name", default="CoLingo"
    )

    args = parser.parse_args()

    id = None
    if args.path and args.id is None:
        with open(args.path, "r") as f:
            if args.path.endswith(".json"):
                config = json.load(f)
            elif args.path.endswith((".yaml", ".yml")):
                config = yaml.safe_load(f)
            elif args.path.endswith(".toml"):
                config = toml.load(f)
            else:
                raise ValueError(f"Unknown file extension: {args.path}")

        id = wandb.sweep(sweep=config)
    if args.path is None and args.id:
        id = args.id

    if id is None:
        raise ValueError("Either --path or --id must be specified")

    wandb.agent(id, function=sweep, project=args.wandb_project)

import json

import toml
import yaml

import wandb
from src.experiment import run_disc_seq_mlp_exp, run_disc_seq_rnn_exp


def sweep():
    wandb.init()
    config = dict(wandb.config)
    run_disc_seq_mlp_exp(config)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("--id", type=str, help="Sweep ID")
    parser.add_argument(
        "--wandb_project", "-p", type=str, help="Wandb project name", default="CoLingo"
    )

    args = parser.parse_args()

    sweep_path = args.config_path
    with open(sweep_path, "r") as f:
        if sweep_path.endswith(".json"):
            sweep_config = json.load(f)
        elif sweep_path.endswith((".yaml", ".yml")):
            sweep_config = yaml.safe_load(f)
        elif sweep_path.endswith(".toml"):
            sweep_config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {sweep_path}")

    sweep_id = args.id if args.id else wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=sweep, project=args.wandb_project)

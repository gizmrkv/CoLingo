import json
from pprint import pprint

import toml
import yaml

import wandb
from src.experiment.monologue import run_monologue


def sweep_monologue():
    wandb.init()
    config = dict(wandb.config)
    run_monologue(config)


if __name__ == "__main__":
    sweep_path = "config_sweep/single1.toml"
    with open(sweep_path, "r") as f:
        if sweep_path.endswith(".json"):
            sweep_config = json.load(f)
        elif sweep_path.endswith((".yaml", ".yml")):
            sweep_config = yaml.safe_load(f)
        elif sweep_path.endswith(".toml"):
            sweep_config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {sweep_path}")

    sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=sweep_monologue)

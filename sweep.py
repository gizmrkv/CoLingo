import json

import toml
import yaml

import wandb
from src.experiment import run_duologue


def sweep_monologue():
    wandb.init()
    config = dict(wandb.config)
    run_duologue(config)


if __name__ == "__main__":
    # sweep_path = "config_sweep/monologue_hyperparam_search.toml"
    # sweep_path = "config_sweep/length_penalty_search.toml"
    sweep_path = "config_sweep/sender_output_search.toml"
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

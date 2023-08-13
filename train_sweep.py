import argparse
import json
import os

import toml
import yaml

import wandb
from colingo.zoo import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--sweep_path", type=str, help="Path to sweep file", default=None
    )
    parser.add_argument("-i", "--sweep_id", type=str, help="Sweep ID", default=None)
    parser.add_argument(
        "-w", "--wandb_project", type=str, help="Wandb project name", default=None
    )

    args = parser.parse_args()

    path: str | None = args.sweep_path
    id: str | None = args.sweep_id
    project: str | None = args.wandb_project

    if path and id is None:
        ext = os.path.splitext(path)[-1][1:]
        with open(path, "r") as f:
            if ext == "json":
                config = json.load(f)
            elif ext in ("yaml", "yml"):
                config = yaml.safe_load(f)
            elif ext == "toml":
                config = toml.load(f)
            else:
                raise ValueError(f"Unknown file extension: {ext}")

        if project is None:
            project = config["project"]
        id = wandb.sweep(sweep=config, project=project)

    if id is None:
        raise ValueError("Sweep ID is not specified.")

    def sweep() -> None:
        wandb.init()
        train(wandb.config)

    if project is None:
        raise ValueError("Wandb project name is not specified.")
    else:
        wandb.agent(id, function=sweep, project=project)

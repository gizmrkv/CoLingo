import argparse
import json
from pathlib import Path
from typing import Any

import toml
import yaml

import wandb
from colingo.zoo import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--sweep", "-s", action="store_true")
    parser.add_argument("--sweep_id", "-i", type=str, default=None)
    parser.add_argument("--sweep_project", "-w", type=str, default=None)

    args = parser.parse_args()
    path: str | None = args.path
    is_sweep: bool = args.sweep
    id: str | None = args.sweep_id
    project: str | None = args.sweep_project

    if is_sweep:
        sweep(path, id, project)
    else:
        if path is None:
            raise ValueError("Path to sweep file is not specified.")

        p = Path(path)
        if p.is_dir():
            for file in p.glob("*"):
                print(f"Training {file}")
                train(read_config(file))

        if p.is_file():
            train(read_config(p))


def sweep(path: str | None, id: str | None, project: str | None) -> None:
    if path and id is None:
        p = Path(path)
        config = read_config(p)
        if project is None:
            project = config["project"]
        id = wandb.sweep(sweep=config, project=project)

    if id is None:
        raise ValueError("Sweep ID is not specified.")

    def func() -> None:
        wandb.init()
        train(wandb.config)

    if project is None:
        raise ValueError("Wandb project name is not specified.")

    wandb.agent(id, function=func, project=project)


def read_config(path: Path) -> dict[str, Any]:
    ext = path.suffix[1:]
    with path.open() as f:
        if ext == "json":
            config = json.load(f)
        elif ext in ("yaml", "yml"):
            config = yaml.safe_load(f)
        elif ext == "toml":
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {path}")
    return config


if __name__ == "__main__":
    main()

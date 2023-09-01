import argparse
import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import toml
import yaml

import wandb
from colingo.utils import fix_seed
from colingo.zoo.reco_network import train_mlp_rnn as train_mlp_rnn_net
from colingo.zoo.reco_network import train_mlp_transformer as train_mlp_transformer_net
from colingo.zoo.reco_signaling import train_mlp_rnn, train_mlp_transformer
from colingo.zoo.reconstruction import train_reconstruction_from_config


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


def train(config: Mapping[str, Any]) -> None:
    seed: int | None = config.get("seed", None)
    target: str = config["zoo"]

    now = datetime.datetime.now()
    log_dir = Path(
        "log",
        now.date().strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S-%f"),
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    with log_dir.joinpath("config.json").open("w") as f:
        json.dump({k: v for k, v in config.items()}, f, indent=4)

    if seed is not None:
        fix_seed(seed)

    if target == "reconstruction":
        train_reconstruction_from_config(config)
    elif target == "reco_signaling_mlp_rnn":
        train_mlp_rnn(config, log_dir)
    elif target == "reco_signaling_mlp_transformer":
        train_mlp_transformer(config, log_dir)
    elif target == "reco_network_mlp_rnn":
        train_mlp_rnn_net(config, log_dir)
    elif target == "reco_network_mlp_transformer":
        train_mlp_transformer_net(config, log_dir)
    else:
        raise ValueError(f"Unknown target: {target}")


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

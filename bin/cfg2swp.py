import argparse
import json
import os
from pathlib import Path

import toml
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to config file")
    parser.add_argument(
        "--ext", "-e", type=str, default="toml", help="Sweep file extension"
    )

    args = parser.parse_args()

    path = Path(args.path)
    ext = path.suffix[1:]

    with path.open() as f:
        if ext == "json":
            config = json.load(f)
        elif ext in ("yaml", "yml"):
            config = yaml.safe_load(f)
        elif ext == "toml":
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {ext}")

    sweep_config = {
        "method": "grid|random|bayes",
        "name": "######",
        "project": config["wandb_project"],
        "metric": {
            "name": "######",
            "goal": "maximize|minimize",
        },
        "parameters": {key: {"value": value} for key, value in config.items()},
    }

    sweep_dir = path.parent.joinpath("sweep")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with sweep_dir.joinpath(f"{path.stem}.{args.ext}").open("w") as f:
        if args.ext == "json":
            json.dump(sweep_config, f, indent=4)
        elif args.ext == "yaml":
            yaml.safe_dump(sweep_config, f)
        elif args.ext == "toml":
            toml.dump(sweep_config, f)
        else:
            raise ValueError(f"Unknown file extension: {args.ext}")

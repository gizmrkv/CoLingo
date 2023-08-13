import argparse
import json
import os

import toml
import yaml

from colingo.zoo import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file")

    args = parser.parse_args()
    path: str = args.config
    ext: str = os.path.splitext(path)[-1][1:]

    with open(path, "r") as f:
        if ext == "json":
            config = json.load(f)
        elif ext in ("yaml", "yml"):
            config = yaml.safe_load(f)
        elif ext == "toml":
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {path}")

    train(config)

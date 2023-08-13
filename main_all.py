import argparse
import json
import os

import toml
import yaml

from colingo.zoo import train

if __name__ == "__main__":
    for root, dirs, files in os.walk(top="./config"):
        if not root.endswith("sweep"):
            for file in files:
                path = os.path.join(root, file)
                ext = os.path.splitext(path)[-1][1:]

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

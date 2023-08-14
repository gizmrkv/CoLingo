import argparse
import json
import os

import toml
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to config file")
    parser.add_argument(
        "dest_ext", type=str, default="toml", help="Sweep file extension"
    )

    args = parser.parse_args()

    file_dir: str = os.path.dirname(args.file_path)
    file_base: str = os.path.splitext(os.path.basename(args.file_path))[0]
    file_ext: str = os.path.splitext(args.file_path)[-1][1:]

    if file_ext not in ("json", "yaml", "yml", "toml"):
        raise ValueError(f"Unknown config file extension: {file_ext}")

    with open(args.file_path, "r") as f:
        if file_ext == "json":
            config = json.load(f)
        elif file_ext in ("yaml", "yml"):
            config = yaml.safe_load(f)
        elif file_ext == "toml":
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {file_ext}")

    sweep_config = {
        "zoo": config["zoo"],
        "method": "grid|random|bayes",
        "name": "######",
        "project": "######",
        "metric": {
            "name": "######",
            "goal": "maximize|minimize",
        },
        "parameters": {key: {"value": value} for key, value in config.items()},
    }

    sweep_dir = os.path.join(file_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    with open(f"{sweep_dir}/{file_base}.{args.dest_ext}", "w") as f:
        if args.dest_ext == "json":
            json.dump(sweep_config, f, indent=4)
        elif args.dest_ext == "yaml":
            yaml.safe_dump(sweep_config, f)
        elif args.dest_ext == "toml":
            toml.dump(sweep_config, f)
        else:
            raise ValueError(f"Unknown file extension: {args.dest_ext}")

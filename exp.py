import json

import toml
import yaml

from src.experiment.single import run_single_experiment

if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "config/single1.toml"
    # config_path = "config/small.yaml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    run_single_experiment(config)

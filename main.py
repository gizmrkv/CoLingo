import json

import toml
import yaml

from src.experiment import run_echoing, run_inferring, run_multilogue

if __name__ == "__main__":
    # config_path = sys.argv[1]
    # config_path = "config/multilogue.toml"
    # config_path = "config/inferring.toml"
    config_path = "config/echoing.toml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    # run_multilogue(config)
    # run_inferring(config)
    run_echoing(config)

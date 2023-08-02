import json

import toml
import yaml

if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "config/single1.toml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    sweep_cfg = {
        "method": "grid|random|bayes",
        "name": "######",
        "project": "######",
        "metric": {
            "name": "######",
            "goal": "maximize|minimize",
        },
        "parameters": {key: {"value": value} for key, value in config.items()},
    }

    with open("config_sweep/single1.toml", "w") as f:
        toml.dump(sweep_cfg, f)

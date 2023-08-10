import json
import os

import toml
import yaml

if __name__ == "__main__":
    # config_path = sys.argv[1]
    config_path = "config/signaling/with_mlp_rnn/1.toml"
    config_path = "config/inferring/with_transformer/1.toml"

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

    os.makedirs("config_sweep/inferring/with_transformer", exist_ok=True)
    with open("config_sweep/inferring/with_transformer/1.toml", "w") as f:
        toml.dump(sweep_cfg, f)

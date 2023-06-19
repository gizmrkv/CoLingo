import json

import toml
import yaml

from src.experiment import run_echoing, run_inferring, run_multilogue, run_signaling

if __name__ == "__main__":
    exp = "signaling"

    # config_path = sys.argv[1]
    config_path = f"config/{exp}.toml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    if exp == "echoing":
        run_echoing(config)
    elif exp == "inferring":
        run_inferring(config)
    elif exp == "signaling":
        run_signaling(config)
    elif exp == "multilogue":
        run_multilogue(config)

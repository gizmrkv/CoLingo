import json

import toml
import yaml

from colingo.zoo.inferring import (
    ConfigWithMLP,
    ConfigWithRNN,
    train_with_mlp,
    train_with_rnn,
)
from colingo.zoo.signaling import ConfigWithMLPRNN, train_with_mlp_rnn

if __name__ == "__main__":
    path = "config/inferring/with_rnn/1.toml"
    path = "config/inferring/with_mlp/1.toml"
    path = "config/signaling/with_mlp_rnn/1.toml"

    # config_path = sys.argv[1]

    with open(path, "r") as f:
        if path.endswith(".json"):
            config = json.load(f)
        elif path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {path}")

    if "inferring/with_mlp" in path:
        train_with_mlp(ConfigWithMLP(**config))
    elif "inferring/with_rnn" in path:
        train_with_rnn(ConfigWithRNN(**config))
    elif "signaling/with_mlp_rnn" in path:
        train_with_mlp_rnn(ConfigWithMLPRNN(**config))

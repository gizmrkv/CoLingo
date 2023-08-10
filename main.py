import json

import toml
import yaml

from colingo.zoo.int_sequence_reco_signaling import train_mlp_rnn, train_mlp_transformer
from colingo.zoo.int_sequence_reconstruction import (
    train_mlp,
    train_rnn,
    train_transformer,
)

if __name__ == "__main__":
    zoo = "int_sequence_reconstruction"
    zoo = "int_sequence_reco_signaling"
    sub = "mlp1"
    sub = "rnn1"
    sub = "transformer1"
    sub = "mlp_rnn1"
    sub = "mlp_transformer1"

    path = f"config/{zoo}/{sub}.toml"

    with open(path, "r") as f:
        if path.endswith(".json"):
            config = json.load(f)
        elif path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {path}")

    if "int_sequence_reconstruction" in zoo:
        if "mlp" in sub:
            train_mlp(config)
        elif "rnn" in sub:
            train_rnn(config)
        elif "transformer" in sub:
            train_transformer(config)
    elif "int_sequence_reco_signaling" in zoo:
        if "mlp_rnn" in sub:
            train_mlp_rnn(config)
        elif "mlp_transformer" in sub:
            train_mlp_transformer(config)

from typing import Any

from .int_sequence_reco_signaling import train_mlp_rnn, train_mlp_transformer
from .int_sequence_reconstruction import train_mlp, train_rnn, train_transformer


def train(config: dict[str, Any]) -> None:
    target: str = config["target"]

    if target == "int_sequence_reconstruction/mlp":
        train_mlp(config)
    elif target == "int_sequence_reconstruction/rnn":
        train_rnn(config)
    elif target == "int_sequence_reconstruction/transformer":
        train_transformer(config)
    elif target == "int_sequence_reco_signaling/mlp_rnn":
        train_mlp_rnn(config)
    elif target == "int_sequence_reco_signaling/mlp_transformer":
        train_mlp_transformer(config)
    else:
        raise ValueError(f"Unknown target: {target}")

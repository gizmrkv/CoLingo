from typing import Any, Mapping

from .int_sequence_reco_signaling import train_mlp_rnn, train_mlp_transformer
from .int_sequence_reconstruction import train_mlp, train_rnn, train_transformer


def train(config: Mapping[str, Any]) -> None:
    target: str = config["zoo"]

    if target == "int_sequence_reconstruction_mlp":
        train_mlp(config)
    elif target == "int_sequence_reconstruction_rnn":
        train_rnn(config)
    elif target == "int_sequence_reconstruction_transformer":
        train_transformer(config)
    elif target == "int_sequence_reco_signaling_mlp_rnn":
        train_mlp_rnn(config)
    elif target == "int_sequence_reco_signaling_mlp_transformer":
        train_mlp_transformer(config)
    else:
        raise ValueError(f"Unknown target: {target}")

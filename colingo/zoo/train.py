from typing import Any, Mapping

from .reco_network import train_mlp_rnn as train_mlp_rnn_net
from .reco_signaling import train_mlp_rnn, train_mlp_transformer
from .reconstruction import train_mlp, train_rnn, train_transformer


def train(config: Mapping[str, Any]) -> None:
    target: str = config["zoo"]

    if target == "reconstruction_mlp":
        train_mlp(config)
    elif target == "reconstruction_rnn":
        train_rnn(config)
    elif target == "reconstruction_transformer":
        train_transformer(config)
    elif target == "reco_signaling_mlp_rnn":
        train_mlp_rnn(config)
    elif target == "reco_signaling_mlp_transformer":
        train_mlp_transformer(config)
    elif target == "reco_network_mlp_rnn":
        train_mlp_rnn_net(config)
    else:
        raise ValueError(f"Unknown target: {target}")

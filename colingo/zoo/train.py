import datetime
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from ..utils import fix_seed
from .reco_network import train_mlp_rnn as train_mlp_rnn_net
from .reco_signaling import train_mlp_rnn, train_mlp_transformer
from .reconstruction import train_mlp, train_rnn, train_transformer


def train(config: Mapping[str, Any]) -> None:
    seed: int | None = config.get("seed", None)
    target: str = config["zoo"]

    now = datetime.datetime.now()
    log_dir = Path(
        "log",
        now.date().strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S-%f"),
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    with log_dir.joinpath("config.json").open("w") as f:
        json.dump({k: v for k, v in config.items()}, f, indent=4)

    if seed is not None:
        fix_seed(seed)

    if target == "reconstruction_mlp":
        train_mlp(config)
    elif target == "reconstruction_rnn":
        train_rnn(config)
    elif target == "reconstruction_transformer":
        train_transformer(config)
    elif target == "reco_signaling_mlp_rnn":
        train_mlp_rnn(config, log_dir)
    elif target == "reco_signaling_mlp_transformer":
        train_mlp_transformer(config, log_dir)
    elif target == "reco_network_mlp_rnn":
        train_mlp_rnn_net(config, log_dir)
    else:
        raise ValueError(f"Unknown target: {target}")

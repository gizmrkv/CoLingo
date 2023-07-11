import json

import toml
import yaml

from src.experiment.disc_seq_inferring.mlp_train import main as mlp_inferring_train
from src.experiment.disc_seq_inferring.rnn_train import main as rnn_inferring_train
from src.experiment.disc_seq_signaling.mlp_rnn_train import (
    main as mlp_rnn_signaling_train,
)

if __name__ == "__main__":
    exp = "disc_seq_inferring/mlp1"
    exp = "disc_seq_inferring/rnn1"
    exp = "disc_seq_signaling/mlp_rnn1"

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

    if exp.startswith("disc_seq_inferring/mlp"):
        mlp_inferring_train(config)
    elif exp.startswith("disc_seq_inferring/rnn"):
        rnn_inferring_train(config)
    elif exp.startswith("disc_seq_signaling/mlp_rnn"):
        mlp_rnn_signaling_train(config)

import json

import toml
import yaml

from src.experiment import (
    run_disc_seq_mlp_exp,
    run_disc_seq_rnn_exp,
    run_mlp_rnn_signaling_exp,
)

if __name__ == "__main__":
    exp = "disc_seq_mlp_exp"
    exp = "disc_seq_rnn_exp"
    exp = "mlp_rnn_signaling_exp"

    # config_path = sys.argv[1]
    config_path = f"config/{exp}/1.toml"

    with open(config_path, "r") as f:
        if config_path.endswith(".json"):
            config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        elif config_path.endswith(".toml"):
            config = toml.load(f)
        else:
            raise ValueError(f"Unknown file extension: {config_path}")

    if exp == "disc_seq_mlp_exp":
        run_disc_seq_mlp_exp(config)
    elif exp == "disc_seq_rnn_exp":
        run_disc_seq_rnn_exp(config)
    elif exp == "mlp_rnn_signaling_exp":
        run_mlp_rnn_signaling_exp(config)

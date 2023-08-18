# CoLingo: Cooperation, Communication and Consensus Language Emergence

## Introduction
CoLingo is a framework for studying language emergence in multi-agent environments. 

## List of implemented games
List of example and tutorial games
 * `colingo/zoo/int_sequence_reconstruction`: Int Sequence Reconstruction;
 * `colingo/zoo/int_sequence_reco_signaling`: Int Sequence Reconstruction with Signaling
 * `colingo/zoo/int_sequence_reco_net_signaling`: Int Sequence Reconstruction with Signaling and Network

## Installing
```
git clone git@github.com:gizmrkv/CoLingo.git
cd CoLingo
pyenv local 3.10.12
pip install -U pip
pip install PyYAML gym isort matplotlib moviepy mypy numpy pandas plotly rapidfuzz scipy seaborn toml torch torchtyping tqdm types-PyYAML types-toml types-tqdm typing-extensions wandb
```

## Structure
The repo is organized as follows:
```
- config   # config files
- colingo
-- core    # common components: runner, trainer, evaluator, ...
-- game    # game implementations
-- module  # module implementations
-- zoo     # experiments
```

## Running
Run one experiment:
```
python -m train_one CONFIG_FILE
``` 

Run all experiments in the config directory:
```
python -m train_all
```

Run new sweep using Weights & Biases:
```
python -m train_sweep -p CONFIG_FILE
```

Continue a sweep that has already been generated:
```
python -m train_sweep -i SWEEP_ID -w WANDB_PROJECT
```
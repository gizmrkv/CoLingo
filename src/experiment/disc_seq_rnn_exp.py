import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from ..baseline import BatchMeanBaseline
from ..core import Runner, fix_seed
from ..dataset import random_split
from ..game import (
    InferringGame,
    InferringGameEvaluator,
    InferringGameResult,
    InferringGameTrainer,
)
from ..logger import EarlyStopper, StepCounter, WandBLogger
from ..loss import DiscSeqReinforceLoss
from ..model import DiscSeqRNNDecoder, DiscSeqRNNEncoder


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str
    exp_name: str
    wandb_project: str
    use_tqdm: bool

    # common config
    lr: float
    length: int
    n_values: int
    latent_dim: int
    reinforce: bool
    baseline: str
    entropy_weight: float

    # encoder config
    encoder_hidden_dim: int
    encoder_embed_dim: int
    encoder_rnn_type: str
    encoder_n_layers: int

    # decoder config
    decoder_hidden_dim: int
    decoder_embed_dim: int
    decoder_rnn_type: str
    decoder_n_layers: int


class Agent(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        input: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
        command: str | None = None,
    ):
        match command:
            case "input":
                return self.encoder(input)
            case "output":
                return self.decoder(latent)
            case _:
                raise ValueError(f"Unknown command: {command}")


def run_disc_seq_rnn_exp(config: dict):
    # pre process
    cfg: Config = Config(**config)

    assert cfg.device in ["cpu", "cuda"], "Invalid device"
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Use CPU instead.")
        cfg.device = "cpu"

    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

    # model
    encoder = DiscSeqRNNEncoder(
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        embed_dim=cfg.encoder_embed_dim,
        rnn_type=cfg.encoder_rnn_type,
        n_layers=cfg.encoder_n_layers,
    )
    decoder = DiscSeqRNNDecoder(
        length=cfg.length,
        n_values=cfg.n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        embed_dim=cfg.decoder_embed_dim,
        rnn_type=cfg.decoder_rnn_type,
        n_layers=cfg.decoder_n_layers,
    )
    agent = Agent(encoder, decoder).to(cfg.device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)

    agents = {"A": agent}
    optimizers = {"A": optimizer}

    # data
    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # game
    if cfg.reinforce:
        baselines = {"batch_mean": BatchMeanBaseline()}
        reinforce_loss = DiscSeqReinforceLoss(
            entropy_weight=cfg.entropy_weight, baseline=baselines[cfg.baseline]
        )

    def loss(result: InferringGameResult, target: torch.Tensor):
        if cfg.reinforce:
            acc = (result.output == result.input).float().mean(dim=-1)
            distr = Categorical(logits=result.info)
            log_prob = distr.log_prob(result.output)
            entropy = distr.entropy()
            return reinforce_loss(acc, log_prob, entropy)
        else:
            logits = result.info.view(-1, cfg.n_values)
            target = target.view(-1)
            return F.cross_entropy(logits, target)

    game = InferringGame()
    trainer = InferringGameTrainer(
        game=game,
        agents=agents,
        optimizers=optimizers,
        dataloader=train_dataloader,
        loss=loss,
    )

    # eval
    def metric(result: InferringGameResult):
        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        met = {"acc_comp": acc_comp, "acc_part": acc_part}
        met |= {f"acc{i}": a for i, a in enumerate(list(acc))}
        return met

    wandb_logger = WandBLogger(project=cfg.wandb_project)
    early_stopper = EarlyStopper(metric="valid.A.acc_comp", threshold=1 - 1e-6)

    train_eval = InferringGameEvaluator(
        game=game,
        agents=agents,
        input=train_dataset,
        metric=metric,
        logger=wandb_logger,
        name="train",
    )
    valid_eval = InferringGameEvaluator(
        game=game,
        agents=agents,
        input=valid_dataset,
        metric=metric,
        logger=[wandb_logger, early_stopper],
        name="valid",
    )

    runner = Runner(
        trainer,
        train_eval,
        valid_eval,
        StepCounter(wandb_logger),
        wandb_logger,
        early_stop=early_stopper,
        use_tqdm=cfg.use_tqdm,
    )
    runner.run(cfg.n_epochs)

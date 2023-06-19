import datetime
import json
import os
import uuid
from dataclasses import dataclass

import torch as th
from torch.utils.data import DataLoader, TensorDataset

from ..agent import ConceptOrMessageAgent
from ..analysis import language_similarity
from ..core.runner import Runner
from ..dataset import random_split
from ..game import InferringGameEvaluator, InferringGameResult, InferringGameTrainer
from ..logger import WandBLogger
from ..scheduler import IntervalScheduler
from ..util import ModelInitializer, ModelSaver, fix_seed


@dataclass
class Config:
    exp_name: str
    n_iterations: int
    batch_size: int
    seed: int
    device: str
    model_save_interval: int
    data_size: int

    # wandb
    wandb_project: str
    wandb_name: str

    # channel
    max_len: int
    vocab_size: int

    # model
    internal_dim: int
    concept_embed_dim: int
    concept_hidden_dim: int
    message_embed_dim: int
    rnn_type: str
    message_hidden_dim: int
    n_layers: int
    share_message_embedding: bool

    # optimizer
    optimizer: str
    lr: float

    # dataset
    split_ratio: float

    # concept
    n_attributes: int = 1
    n_values: int = 1


def run_echoing(config: dict):
    # make config
    cfg = Config(**config)

    # check device
    assert cfg.device in ["cpu", "cuda"]
    # assert cfg.device == "cpu" or th.cuda.is_available()

    # make log dir
    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save config
    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

    # make dataset
    dataset = th.randint(0, cfg.vocab_size, (cfg.data_size, cfg.max_len)).to(cfg.device)
    train_dataset, valid_dataset = random_split(
        dataset, [cfg.split_ratio, 1 - cfg.split_ratio]
    )
    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # make model
    agents = {
        "A": ConceptOrMessageAgent(
            n_attributes=cfg.n_attributes,
            n_values=cfg.n_values,
            max_len=cfg.max_len,
            vocab_size=cfg.vocab_size,
            internal_dim=cfg.internal_dim,
            concept_embed_dim=cfg.concept_embed_dim,
            concept_hidden_dim=cfg.concept_hidden_dim,
            message_embed_dim=cfg.message_embed_dim,
            rnn_type=cfg.rnn_type,
            message_hidden_dim=cfg.message_hidden_dim,
            n_layers=cfg.n_layers,
            share_message_embedding=cfg.share_message_embedding,
        ).to(cfg.device)
    }

    # make optimizer
    optimizer_types = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizers = {
        "A": optimizer_types[cfg.optimizer](agents["A"].parameters(), lr=cfg.lr)
    }

    # make agent saver
    agent_saver = ModelSaver(
        models=agents,
        path=f"{log_dir}/models",
    )

    # make agent initializer
    agent_initializer = ModelInitializer(
        model=agents.values(),
    )

    # make loggers
    loggers = [
        WandBLogger(project=cfg.wandb_project, name=cfg.wandb_name),
    ]

    # make concept loss
    def sequence_loss(output: th.Tensor, target: th.Tensor):
        output = output.view(-1, cfg.vocab_size)
        target = target.view(-1)
        return th.nn.functional.cross_entropy(output, target)

    trainer = InferringGameTrainer(
        agents=agents,
        optimizers=optimizers,
        dataloader=train_dataloader,
        loss=sequence_loss,
        input_command="echo_input",
        output_command="echo",
    )

    def inferring_metric(result: InferringGameResult):
        output = result.output.argmax(dim=-1).cpu().numpy()
        target = result.target.cpu().numpy()
        lansim = language_similarity(output, target)
        return {"lansim": lansim}

    evaluators = [
        InferringGameEvaluator(
            agents=agents,
            input=dataset,
            target=dataset,
            metric=inferring_metric,
            logger=loggers,
            name=name,
            input_command="echo_input",
            output_command="echo",
        )
        for dataset, name in [(train_dataset, "train"), (valid_dataset, "valid")]
    ]

    # make callbacks
    callbacks = [
        IntervalScheduler(agent_saver, 1000),
        IntervalScheduler(agent_initializer, 1000),
        IntervalScheduler(trainer, 1),
        IntervalScheduler(evaluators, 1),
    ]
    callbacks.extend(loggers)

    # make runner
    runner = Runner(callbacks)

    # run
    runner.run(n_iterations=cfg.n_iterations)

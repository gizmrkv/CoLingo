import datetime
import json
import os
import uuid
from dataclasses import dataclass

import torch as th
from torch.utils.data import DataLoader, TensorDataset

from ..agent import ConceptOrMessageAgent
from ..baseline import BatchMeanBaseline
from ..core.task_runner import TaskRunner
from ..dataset import concept_dataset, random_split
from ..logger import WandBLogger
from ..loss import ConceptLoss, MessageLoss
from ..metric import ConceptAccuracyMetric, MessageMetric
from ..task.signal import SignalEvaluator, SignalTrainer
from ..task.single import SingleEvaluator, SingleTrainer
from ..util import AgentInitializer, AgentSaver, fix_seed


@dataclass
class Config:
    exp_name: str
    n_iterations: int
    batch_size: int
    seed: int
    device: str
    model_save_interval: int

    # wandb
    wandb_project: str
    wandb_name: str

    # concept
    n_attributes: int
    n_values: int

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

    # reinforce
    baseline: str
    entropy_weight: float
    length_weight: float

    # task
    run_single: bool
    run_signal: bool
    max_batches_single: int
    max_batches_signal: int

    # agent
    agent_name: str = "A1"


def run_monologue(config: dict):
    cfg = Config(**config)

    now = datetime.datetime.now()
    log_id = str(uuid.uuid4())[-4:]
    log_dir = f"log/{cfg.exp_name}_{now.date()}_{now.strftime('%H%M%S')}_{log_id}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(f"{log_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    fix_seed(cfg.seed)

    dataset = concept_dataset(cfg.n_attributes, cfg.n_values, device=cfg.device)
    train_dataset, valid_dataset = random_split(
        dataset, [cfg.split_ratio, 1 - cfg.split_ratio]
    )
    train_dataloader = DataLoader(
        TensorDataset(train_dataset, train_dataset),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    # TODO: Fix
    agent = ConceptOrMessageAgent(
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

    optimizers = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizer = optimizers[cfg.optimizer](agent.parameters(), lr=cfg.lr)

    tasks = [
        AgentSaver(
            agents={cfg.agent_name: agent},
            interval=cfg.model_save_interval,
            path=f"{log_dir}/models",
        ),
        AgentInitializer(
            agents=[agent],
        ),
    ]
    loggers = [
        WandBLogger(project=cfg.wandb_project, name=cfg.wandb_name),
    ]
    tasks.extend(loggers)

    if cfg.run_single:
        single_metrics = [
            ConceptAccuracyMetric(cfg.n_attributes, cfg.n_values),
        ]
        tasks.extend(
            [
                SingleTrainer(
                    agents={cfg.agent_name: agent},
                    optimizers={cfg.agent_name: optimizer},
                    dataloader=train_dataloader,
                    loss=ConceptLoss(cfg.n_attributes, cfg.n_values).to(cfg.device),
                    input_key=0,
                    output_key=0,
                    max_batches=cfg.max_batches_single,
                ),
                SingleEvaluator(
                    agents={cfg.agent_name: agent},
                    input=train_dataset,
                    target=train_dataset,
                    metrics=single_metrics,
                    loggers=loggers,
                    input_key=0,
                    output_key=0,
                    name="single_train",
                ),
                SingleEvaluator(
                    agents={cfg.agent_name: agent},
                    input=valid_dataset,
                    target=valid_dataset,
                    metrics=single_metrics,
                    loggers=loggers,
                    input_key=0,
                    output_key=0,
                    name="single_valid",
                ),
            ]
        )
    if cfg.run_signal:
        baselines = {"batch_mean": BatchMeanBaseline}
        baseline = baselines[cfg.baseline]()
        length_baseline = baselines[cfg.baseline]()

        signal_metrics = [
            ConceptAccuracyMetric(cfg.n_attributes, cfg.n_values),
            MessageMetric(),
        ]

        tasks.extend(
            [
                SignalTrainer(
                    agents={cfg.agent_name: agent},
                    optimizers={cfg.agent_name: optimizer},
                    dataloader=train_dataloader,
                    sender_loss=MessageLoss(
                        entropy_weight=cfg.entropy_weight,
                        length_weight=cfg.length_weight,
                        baseline=baseline,
                        length_baseline=length_baseline,
                    ).to(cfg.device),
                    receiver_loss=ConceptLoss(cfg.n_attributes, cfg.n_values).to(
                        cfg.device
                    ),
                    sender_input_key=0,
                    sender_output_key=1,
                    receiver_input_key=1,
                    receiver_output_key=0,
                    max_batches=cfg.max_batches_signal,
                    channels=[(cfg.agent_name, cfg.agent_name)],
                ),
                SignalEvaluator(
                    agents={cfg.agent_name: agent},
                    input=train_dataset,
                    target=train_dataset,
                    metrics=signal_metrics,
                    loggers=loggers,
                    sender_input_key=0,
                    sender_output_key=1,
                    receiver_input_key=1,
                    receiver_output_key=0,
                    name="signal_train",
                    channels=[(cfg.agent_name, cfg.agent_name)],
                ),
                SignalEvaluator(
                    agents={cfg.agent_name: agent},
                    input=valid_dataset,
                    target=valid_dataset,
                    metrics=signal_metrics,
                    loggers=loggers,
                    sender_input_key=0,
                    sender_output_key=1,
                    receiver_input_key=1,
                    receiver_output_key=0,
                    name="signal_valid",
                    channels=[(cfg.agent_name, cfg.agent_name)],
                ),
            ]
        )

    runner = TaskRunner(tasks)
    runner.run(n_iterations=cfg.n_iterations)

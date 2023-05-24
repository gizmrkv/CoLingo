import datetime
import json
import os
import uuid
from dataclasses import dataclass

import torch as th
from torch.utils.data import DataLoader

from ..core.agent import Agent
from ..core.baseline import BatchMeanBaseline
from ..core.dataset import generate_concept_dataset, random_split
from ..core.logger import ConsoleLogger, WandBLogger
from ..core.loss import ConceptLoss, ReinforceLoss
from ..core.metric import ConceptAccuracy
from ..core.network import generate_custom_graph
from ..core.task_runner import TaskRunner
from ..core.util import ModelInitializer, ModelSaver, fix_seed
from ..model.cross_modal import CrossModalModel
from ..task.signal import SignalEvaluator, SignalTrainer
from ..task.single import SingleEvaluator, SingleTrainer


@dataclass
class Config:
    exp_name: str
    n_iterations: int
    batch_size: int
    seed: int
    device: str
    model_save_interval: int

    # concept
    n_attributes: int
    n_values: int

    # channel
    max_len: int
    vocab_size: int

    # model
    internal_size: int
    embed_size: int
    hidden_size: int
    rnn_type: str
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

    dataset = generate_concept_dataset(cfg.n_attributes, cfg.n_values)
    train_dataset, valid_dataset = random_split(
        dataset, [cfg.split_ratio, 1 - cfg.split_ratio]
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    model = CrossModalModel(
        n_attributes=cfg.n_attributes,
        n_values=cfg.n_values,
        max_len=cfg.max_len,
        vocab_size=cfg.vocab_size,
        internal_size=cfg.internal_size,
        embed_size=cfg.embed_size,
        hidden_size=cfg.hidden_size,
        rnn_type=cfg.rnn_type,
        n_layers=cfg.n_layers,
        share_message_embedding=cfg.share_message_embedding,
    )
    optimizers = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizer = optimizers[cfg.optimizer](model.parameters(), lr=cfg.lr)
    agent = Agent(model, optimizer)
    tasks = [
        ModelSaver(
            agents={"agent": agent},
            interval=cfg.model_save_interval,
            path=f"{log_dir}/models",
        ),
        ModelInitializer(
            agents={"agent": agent},
            network=generate_custom_graph(nodes=["agent"]),
        ),
    ]
    if cfg.run_single:
        tasks.append(
            SingleTrainer(
                agents={"agent": agent},
                dataloader=train_dataloader,
                loss=ConceptLoss(cfg.n_attributes, cfg.n_values),
                network=generate_custom_graph(nodes=["agent"]),
                max_batches=cfg.max_batches_single,
            )
        )
        tasks.append(
            SingleEvaluator(
                agents={"agent": agent},
                dataloader=valid_dataloader,
                metrics={"acc": ConceptAccuracy(cfg.n_attributes, cfg.n_values)},
                loggers={"console": ConsoleLogger(), "wandb": WandBLogger()},
                network=generate_custom_graph(nodes=["agent"]),
            )
        )
    if cfg.run_signal:
        baselines = {
            "batch_mean": BatchMeanBaseline,
        }
        baseline = baselines[cfg.baseline]()
        length_baseline = baselines[cfg.baseline]()
        tasks.append(
            SignalTrainer(
                agents={"agent": agent},
                dataloader=train_dataloader,
                sender_loss=ReinforceLoss(
                    entropy_weight=cfg.entropy_weight,
                    length_weight=cfg.length_weight,
                    baseline=baseline,
                    length_baseline=length_baseline,
                ),
                receiver_loss=ConceptLoss(cfg.n_attributes, cfg.n_values),
                network=generate_custom_graph(
                    nodes=["agent"], edges=[("agent", "agent")]
                ),
                max_batches=cfg.max_batches_signal,
            )
        )
        tasks.append(
            SignalEvaluator(
                agents={"agent": agent},
                dataloader=valid_dataloader,
                metrics={"acc": ConceptAccuracy(cfg.n_attributes, cfg.n_values)},
                loggers={"console": ConsoleLogger(), "wandb": WandBLogger()},
                network=generate_custom_graph(
                    nodes=["agent"], edges=[("agent", "agent")]
                ),
            )
        )

    runner = TaskRunner(tasks)
    runner.run(n_iterations=cfg.n_iterations)

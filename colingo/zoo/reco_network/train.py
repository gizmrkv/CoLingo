from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Set

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from networkx import DiGraph
from torch.utils.data import DataLoader
from torchtyping import TensorType

import wandb

from ...core import Computable, Evaluator, Loggable, Task, TaskRunner, Trainer
from ...loggers import HeatmapLogger, Namer, WandbLogger
from ...module import (
    MLPDecoder,
    MLPEncoder,
    RNNDecoder,
    RNNEncoder,
    TransformerDecoder,
    TransformerEncoder,
)
from ...tasks import DictStopper, KeyChecker, StepCounter, Stopwatch, TimeDebugger
from ...utils import init_weights, random_split
from .game import (
    RecoNetworkAgent,
    RecoNetworkGame,
    RecoNetworkGameResult,
    RecoNetworkSubGame,
    RecoNetworkSubGameResult,
)
from .loss import Loss
from .metrics import (
    AccuracyHeatmapMetrics,
    LanguageLoggerWrapper,
    LanguageSimilarityMetrics,
    Metrics,
    TopographicSimilarityMetrics,
)


def train_reco_network(
    agents: Mapping[str, RecoNetworkAgent],
    network: DiGraph,
    concept_length: int,
    concept_values: int,
    train_proportion: float,
    valid_proportion: float,
    message_max_len: int,
    message_vocab_size: int,
    entropy_weight: float,
    length_weight: float,
    receiver_loss_weight: float,
    sender_loss_weight: float,
    receiver_imitation_loss_weight: float,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    wandb_project: str,
    use_tqdm: bool,
    metrics_interval: int,
    topsim_interval: int,
    lang_log_interval: int,
    acc_heatmap_interval: int,
    lansim_interval: int,
    log_dir: Path,
    additional_tasks: Iterable[Task] | None = None,
) -> None:
    optimizers = {
        name: optim.Adam(agent.parameters(), lr=lr) for name, agent in agents.items()
    }

    for agent in agents.values():
        agent.to(device)
        agent.apply(init_weights)

    dataset = (
        torch.Tensor(list(product(torch.arange(concept_values), repeat=concept_length)))
        .long()
        .to(device)
    )
    train_dataset, valid_dataset = random_split(
        dataset, [train_proportion, valid_proportion]
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False
    )

    def baseline(x: TensorType[..., float]) -> TensorType[..., float]:
        return x.detach().mean(dim=0)

    loss = Loss(
        concept_length=concept_length,
        concept_values=concept_values,
        message_max_len=message_max_len,
        message_vocab_size=message_vocab_size,
        entropy_weight=entropy_weight,
        length_weight=length_weight,
        baseline=baseline,
        length_baseline=baseline,
        receiver_loss_weight=receiver_loss_weight,
        sender_loss_weight=sender_loss_weight,
        receiver_imitation_loss_weight=receiver_imitation_loss_weight,
    )

    game = RecoNetworkGame(agents, network)
    trainer = Trainer(
        agents=agents.values(),
        input=train_dataloader,
        game=game,
        loss=loss,
        optimizers=optimizers.values(),
    )

    wandb_logger = WandbLogger(project=wandb_project)
    key_checker = KeyChecker()
    loggers: List[Loggable[Mapping[str, Any]]] = [wandb_logger, key_checker]

    net_comp = nx.complete_graph(agents, nx.DiGraph())
    net_comp.add_edges_from([(a, a) for a in agents])
    game_comp = RecoNetworkGame(agents, net_comp)

    heatmap_option = {
        "vmin": 0,
        "vmax": 1,
        "cmap": "viridis",
        "annot": True,
        "fmt": ".2f",
        "cbar": True,
        "square": True,
    }

    class WandbHeatmapLogger(Loggable[Path]):
        def __init__(self, loggers: Iterable[Loggable[Mapping[str, Any]]]) -> None:
            self.loggers = loggers

        def log(self, path: Path) -> None:
            for logger in self.loggers:
                logger.log({"heatmap": wandb.Video(path.as_posix())})

    evaluators = []
    heatmap_loggers = []
    for name, input in [
        ("train", train_dataloader),
        ("valid", valid_dataloader),
    ]:
        metrics = Metrics(loss=loss, loggers=[Namer(name, loggers)])
        topsim_metrics = TopographicSimilarityMetrics(loggers=[Namer(name, loggers)])

        acc_comp_heatmap_logger = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_acc_comp"),
            heatmap_option=heatmap_option,
            loggers=[
                WandbHeatmapLogger(loggers=[Namer(f"{name}.acc_comp", loggers)]),
            ],
        )
        acc_part_heatmap_logger = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_acc_part"),
            heatmap_option=heatmap_option,
            loggers=[
                WandbHeatmapLogger(loggers=[Namer(f"{name}.acc_part", loggers)]),
            ],
        )

        acc_heatmap_metrics = AccuracyHeatmapMetrics(
            acc_comp_heatmap_logger=acc_comp_heatmap_logger,
            acc_part_heatmap_logger=acc_part_heatmap_logger,
        )

        lansim_heatmap_logger = HeatmapLogger(
            save_dir=log_dir.joinpath(f"{name}_lansim"),
            heatmap_option=heatmap_option,
            loggers=[
                WandbHeatmapLogger(loggers=[Namer(f"{name}.lansim", loggers)]),
            ],
        )
        lansim_metrics = LanguageSimilarityMetrics(
            loggers=[Namer(f"{name}.lansim", loggers)],
            heatmap_logger=lansim_heatmap_logger,
        )

        heatmap_loggers.extend(
            [acc_comp_heatmap_logger, acc_part_heatmap_logger, lansim_heatmap_logger]
        )

        evaluators.append(
            Evaluator(
                agents=agents.values(),
                input=input,
                game=game_comp,
                metrics=[metrics, topsim_metrics, acc_heatmap_metrics, lansim_metrics],
                intervals=[
                    metrics_interval,
                    topsim_interval,
                    acc_heatmap_interval,
                    lansim_interval,
                ],
            )
        )

    net_none = nx.DiGraph()
    net_none.add_nodes_from(agents)
    game_none = RecoNetworkGame(agents, net_none)

    language_logger = LanguageLoggerWrapper(log_dir.joinpath("lang"), agents)
    evaluators.append(
        Evaluator(
            agents=agents.values(),
            input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
            game=game_none,
            metrics=[language_logger],
            intervals=[lang_log_interval],
        )
    )

    runner_callbacks = [
        *(additional_tasks or []),
        trainer,
        *evaluators,
        StepCounter(loggers),
        Stopwatch(loggers),
        *heatmap_loggers,
        wandb_logger,
        key_checker,
    ]
    # runner_callbacks = [TimeDebugger(runner_callbacks)]
    runner = TaskRunner(runner_callbacks, use_tqdm=use_tqdm)
    runner.run(n_epochs)


@dataclass
class RecoNetworkConfig:
    concept_length: int
    concept_values: int
    train_proportion: float
    valid_proportion: float
    message_max_len: int
    message_vocab_size: int
    entropy_weight: float
    length_weight: float
    receiver_loss_weight: float
    sender_loss_weight: float
    receiver_imitation_loss_weight: float
    n_epochs: int
    batch_size: int
    lr: float
    device: str
    wandb_project: str
    use_tqdm: bool
    metrics_interval: int
    topsim_interval: int
    lang_log_interval: int
    acc_heatmap_interval: int
    lansim_interval: int

    network_type: Literal["complete", "ring"]

    agents_mode: Literal["diverse", "uniform"]

    network_params: Mapping[str, Any] | None = None

    # for diverse
    agent_configs: Mapping[str, Mapping[str, Any]] | None = None

    # for uniform
    n_agents: int | None = None
    agent_config: Mapping[str, Any] | None = None


def train_reco_network_from_config(
    config: Mapping[str, Any],
    log_dir: Path,
    additional_tasks: Iterable[Task] | None = None,
) -> None:
    fields = RecoNetworkConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = RecoNetworkConfig(**config)

    if cfg.agents_mode == "diverse":
        if cfg.agent_configs is None:
            raise ValueError("agent_configs must be given")

        agents = {
            name: generate_agent(
                concept_length=cfg.concept_length,
                concept_values=cfg.concept_values,
                message_max_len=cfg.message_max_len,
                message_vocab_size=cfg.message_vocab_size,
                config=config,
            )
            for name, config in cfg.agent_configs.items()
        }
    elif cfg.agents_mode == "uniform":
        if cfg.n_agents is None or cfg.agent_config is None:
            raise ValueError("n_agents and agent_config must be given")

        agent = generate_agent(
            concept_length=cfg.concept_length,
            concept_values=cfg.concept_values,
            message_max_len=cfg.message_max_len,
            message_vocab_size=cfg.message_vocab_size,
            config=cfg.agent_config,
        )
        agents = {f"A{i}": deepcopy(agent) for i in range(cfg.n_agents)}

    if cfg.network_type == "complete":
        network = nx.complete_graph(agents, nx.DiGraph())
        network.add_edges_from([(a, a) for a in agents])
    elif cfg.network_type == "ring":
        network = nx.DiGraph()
        network.add_nodes_from(agents)
        for i in range(len(agents)):
            network.add_edge(f"A{i}", f"A{(i + 1) % len(agents)}")

    train_reco_network(
        agents=agents,
        network=network,
        concept_length=cfg.concept_length,
        concept_values=cfg.concept_values,
        train_proportion=cfg.train_proportion,
        valid_proportion=cfg.valid_proportion,
        message_max_len=cfg.message_max_len,
        message_vocab_size=cfg.message_vocab_size,
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        receiver_loss_weight=cfg.receiver_loss_weight,
        sender_loss_weight=cfg.sender_loss_weight,
        receiver_imitation_loss_weight=cfg.receiver_imitation_loss_weight,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=cfg.device,
        wandb_project=cfg.wandb_project,
        use_tqdm=cfg.use_tqdm,
        metrics_interval=cfg.metrics_interval,
        topsim_interval=cfg.topsim_interval,
        lang_log_interval=cfg.lang_log_interval,
        acc_heatmap_interval=cfg.acc_heatmap_interval,
        lansim_interval=cfg.lansim_interval,
        log_dir=log_dir,
        additional_tasks=additional_tasks,
    )


@dataclass
class RecoNetworkAgentConfig:
    latent_dim: int
    concept_encoder_type: Literal["mlp", "rnn", "transformer"]
    concept_encoder_params: Mapping[str, Any]
    concept_decoder_type: Literal["mlp", "rnn", "transformer"]
    concept_decoder_params: Mapping[str, Any]
    message_encoder_type: Literal["mlp", "rnn", "transformer"]
    message_encoder_params: Mapping[str, Any]
    message_decoder_type: Literal["mlp", "rnn", "transformer"]
    message_decoder_params: Mapping[str, Any]


def generate_agent(
    concept_length: int,
    concept_values: int,
    message_max_len: int,
    message_vocab_size: int,
    config: Mapping[str, Any],
) -> RecoNetworkAgent:
    fields = RecoNetworkAgentConfig.__dataclass_fields__
    config = {k: v for k, v in config.items() if k in fields}
    cfg = RecoNetworkAgentConfig(**config)

    if cfg.concept_encoder_type == "mlp":
        concept_encoder: nn.Module = MLPEncoder(
            max_len=concept_length,
            vocab_size=concept_values,
            output_dim=cfg.latent_dim,
            **cfg.concept_encoder_params,
        )
    elif cfg.concept_encoder_type == "rnn":
        concept_encoder = RNNEncoder(
            vocab_size=concept_values,
            output_dim=cfg.latent_dim,
            **cfg.concept_encoder_params,
        )
    elif cfg.concept_encoder_type == "transformer":
        concept_encoder = TransformerEncoder(
            vocab_size=concept_values,
            output_dim=cfg.latent_dim,
            **cfg.concept_encoder_params,
        )

    if cfg.concept_decoder_type == "mlp":
        concept_decoder: nn.Module = MLPDecoder(
            input_dim=cfg.latent_dim,
            max_len=concept_length,
            vocab_size=concept_values,
            **cfg.concept_decoder_params,
        )
    elif cfg.concept_decoder_type == "rnn":
        concept_decoder = RNNDecoder(
            input_dim=cfg.latent_dim,
            max_len=concept_length,
            vocab_size=concept_values,
            **cfg.concept_decoder_params,
        )
    elif cfg.concept_decoder_type == "transformer":
        concept_decoder = TransformerDecoder(
            input_dim=cfg.latent_dim,
            max_len=concept_length,
            vocab_size=concept_values,
            **cfg.concept_decoder_params,
        )

    if cfg.message_encoder_type == "mlp":
        message_encoder: nn.Module = MLPEncoder(
            max_len=message_max_len,
            vocab_size=message_vocab_size,
            output_dim=cfg.latent_dim,
            **cfg.message_encoder_params,
        )
    elif cfg.message_encoder_type == "rnn":
        message_encoder = RNNEncoder(
            vocab_size=message_vocab_size,
            output_dim=cfg.latent_dim,
            **cfg.message_encoder_params,
        )
    elif cfg.message_encoder_type == "transformer":
        message_encoder = TransformerEncoder(
            vocab_size=message_vocab_size,
            output_dim=cfg.latent_dim,
            **cfg.message_encoder_params,
        )

    if cfg.message_decoder_type == "mlp":
        message_decoder: nn.Module = MLPDecoder(
            input_dim=cfg.latent_dim,
            max_len=message_max_len,
            vocab_size=message_vocab_size,
            **cfg.message_decoder_params,
        )
    elif cfg.message_decoder_type == "rnn":
        message_decoder = RNNDecoder(
            input_dim=cfg.latent_dim,
            max_len=message_max_len,
            vocab_size=message_vocab_size,
            **cfg.message_decoder_params,
        )
    elif cfg.message_decoder_type == "transformer":
        message_decoder = TransformerDecoder(
            input_dim=cfg.latent_dim,
            max_len=message_max_len,
            vocab_size=message_vocab_size,
            **cfg.message_decoder_params,
        )

    return RecoNetworkAgent(
        concept_encoder=concept_encoder,
        concept_decoder=concept_decoder,
        message_encoder=message_encoder,
        message_decoder=message_decoder,
    )

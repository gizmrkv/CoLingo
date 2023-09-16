from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping

import networkx as nx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from networkx import DiGraph
from torch.utils.data import DataLoader
from torchtyping import TensorType

from ...core import Evaluator, Loggable, Task, TaskRunner, Trainer
from ...loggers import (
    HeatmapLogger,
    ImageToVideoTask,
    KeyChecker,
    KeyPrefix,
    KeySuffix,
    LambdaLogger,
    LanguageLogger,
    LanguageSimilarityLogger,
    StepCounter,
    Stopwatch,
    TimeDebugger,
    TopographicSimilarityLogger,
    WandbLogger,
)
from ...module import (
    MLPDecoder,
    MLPEncoder,
    RNNDecoder,
    RNNEncoder,
    TransformerDecoder,
    TransformerEncoder,
)
from ...utils import init_weights, random_split
from .game import RecoNetworkAgent, RecoNetworkGame, RecoNetworkGameResult
from .loss import Loss
from .metrics import AccuracyHeatmapLogger, MetricsLogger


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

    def video_to_wandb(p: Path) -> Dict[str, Any]:
        return {"video": wandb.Video(p.as_posix())}

    evaluators = []
    video_tasks = []
    for name, input in [
        ("train", train_dataloader),
        ("valid", valid_dataloader),
    ]:
        logs = [KeyPrefix(name + ".", loggers)]
        metrics = MetricsLogger(loss, logs)
        topsim = TopographicSimilarityLogger[RecoNetworkGameResult](logs)

        acc_comp_path = log_dir.joinpath(f"{name}_acc_comp")
        acc_comp_heatmap_logger = HeatmapLogger(
            acc_comp_path, heatmap_option=heatmap_option
        )
        video_tasks.append(
            ImageToVideoTask(
                acc_comp_path,
                loggers=[LambdaLogger(video_to_wandb, [KeySuffix(".acc_comp", logs)])],
            )
        )

        acc_part_path = log_dir.joinpath(f"{name}_acc_part")
        acc_part_heatmap_logger = HeatmapLogger(
            acc_part_path, heatmap_option=heatmap_option
        )
        video_tasks.append(
            ImageToVideoTask(
                acc_part_path,
                loggers=[LambdaLogger(video_to_wandb, [KeySuffix(".acc_part", logs)])],
            )
        )

        acc_heatmap_logger = AccuracyHeatmapLogger(
            acc_comp_heatmap_logger=acc_comp_heatmap_logger,
            acc_part_heatmap_logger=acc_part_heatmap_logger,
        )

        langsim_path = log_dir.joinpath(f"{name}_langsim")
        langsim_heatmap_logger = HeatmapLogger(
            langsim_path, heatmap_option=heatmap_option
        )
        video_tasks.append(
            ImageToVideoTask(
                langsim_path,
                loggers=[LambdaLogger(video_to_wandb, [KeySuffix(".langsim", logs)])],
            )
        )
        lansim_logger = LanguageSimilarityLogger[RecoNetworkGameResult](
            list(agents.keys()),
            loggers=logs,
            heatmap_logger=langsim_heatmap_logger,
        )

        evaluators.append(
            Evaluator(
                agents=agents.values(),
                input=input,
                game=game_comp,
                loggers=[metrics, topsim, acc_heatmap_logger, lansim_logger],
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

    def lang_to_wandb(p: Path) -> Dict[str, Any]:
        return {"lang": wandb.Table(dataframe=pd.read_csv(p))}

    lang_logger = LanguageLogger[RecoNetworkGameResult](
        log_dir.joinpath("lang"), [LambdaLogger(lang_to_wandb, loggers)]
    )
    evaluators.append(
        Evaluator(
            agents=agents.values(),
            input=DataLoader(dataset, batch_size=len(dataset)),  # type: ignore
            game=game_none,
            loggers=[lang_logger],
            intervals=[lang_log_interval],
        )
    )

    runner_callbacks = [
        *(additional_tasks or []),
        trainer,
        *evaluators,
        StepCounter(loggers),
        Stopwatch(loggers),
        *video_tasks,
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

    agent_keys = list(agents)
    if cfg.network_type == "complete":
        network = nx.complete_graph(agents, nx.DiGraph())
        network.add_edges_from([(a, a) for a in agents])
    elif cfg.network_type == "ring":
        network = nx.DiGraph()
        network.add_nodes_from(agents)
        for i in range(len(agent_keys)):
            network.add_edge(agent_keys[i], agent_keys[(i + 1) % len(agents)])
    elif cfg.network_type == "line":
        network = nx.DiGraph()
        network.add_nodes_from(agents)
        for i in range(len(agent_keys) - 1):
            network.add_edge(agent_keys[i], agent_keys[i + 1])
            network.add_edge(agent_keys[i + 1], agent_keys[i])

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

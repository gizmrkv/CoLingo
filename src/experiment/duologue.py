import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset

from ..agent import ConceptOrMessageAgent
from ..analysis import (
    LanguageEvaluator,
    concept_accuracy,
    concept_topographic_similarity,
    language_similarity,
    language_uniques,
)
from ..baseline import BatchMeanBaseline
from ..core.task_runner import TaskRunner
from ..dataset import concept_dataset, random_split
from ..game import (
    MessageSignalingGame,
    MessageSignalingGameEvaluator,
    MessageSignalingGameResult,
    MessageSignalingGameTrainer,
)
from ..logger import WandBLogger
from ..loss import ConceptLoss
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
    max_batches_signal: int

    # agent
    agent1_name: str = "A1"
    agent2_name: str = "A2"

    # signal option
    sender_output: bool = False
    receiver_parrot: bool = False


def run_duologue(config: dict):
    cfg = Config(**config)

    assert cfg.device in ["cpu", "cuda"]
    assert cfg.device == "cpu" or th.cuda.is_available()

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
    agents = {
        agent_name: ConceptOrMessageAgent(
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
        for agent_name in [cfg.agent1_name, cfg.agent2_name]
    }

    optimizer_types = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizers = {
        agent_name: optimizer_types[cfg.optimizer](agent.parameters(), lr=cfg.lr)
        for agent_name, agent in agents.items()
    }

    tasks = [
        AgentSaver(
            agents=agents,
            interval=cfg.model_save_interval,
            path=f"{log_dir}/models",
        ),
        AgentInitializer(
            agents=agents.values(),
        ),
    ]
    loggers = [
        WandBLogger(project=cfg.wandb_project, name=cfg.wandb_name),
    ]

    tasks.extend(loggers)

    baselines = {"batch_mean": BatchMeanBaseline}
    loss_baseline = baselines[cfg.baseline]()
    length_baseline = baselines[cfg.baseline]()

    concept_loss = ConceptLoss(n_attributes=cfg.n_attributes, n_values=cfg.n_values).to(
        cfg.device
    )

    game = MessageSignalingGame(
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        loss_baseline=loss_baseline,
        length_baseline=length_baseline,
    ).to(cfg.device)

    game_trainer = MessageSignalingGameTrainer(
        game=game,
        loss=concept_loss,
        agents=agents,
        optimizers=optimizers,
        dataloader=train_dataloader,
    )

    def game_metric(
        result: MessageSignalingGameResult,
        sender_name: str,
        receiver_name: str,
    ):
        output_r = result.receiver_output.argmax(dim=-1)
        acc_part_r, acc_comp_r, acc_r = concept_accuracy(output_r, result.target)
        metrics_s = {
            "log_prob": result.sender_log_prob.mean().item(),
            "entropy": result.sender_entropy.mean().item(),
            "length": result.sender_length.float().mean().item(),
            "uniques": language_uniques(result.sender_message)
            / result.sender_message.shape[0],
        }
        metrics_r = {
            "acc_part": acc_part_r,
            "acc_comp": acc_comp_r,
        }
        metrics_r |= {f"acc_attr{str(i)}": acc for i, acc in enumerate(list(acc_r))}

        return {
            sender_name: metrics_s,
            receiver_name: metrics_r,
        }

    game_train_evaluator = MessageSignalingGameEvaluator(
        game=game,
        agents=agents,
        input=train_dataset,
        target=train_dataset,
        metric=game_metric,
        logger=loggers,
        name="train",
    )
    game_valid_evaluator = MessageSignalingGameEvaluator(
        game=game,
        agents=agents,
        input=valid_dataset,
        target=valid_dataset,
        metric=game_metric,
        logger=loggers,
        name="valid",
    )

    tasks.extend(
        [
            game_trainer,
            game_train_evaluator,
            game_valid_evaluator,
        ]
    )

    def language_metric(
        input: np.ndarray,
        languages: dict[str, np.ndarray],
        lengths: dict[str, np.ndarray],
    ) -> dict:
        topsims = {}
        for agent_name, language in languages.items():
            topsim = concept_topographic_similarity(concept=input, language=language)
            topsims[agent_name] = topsim

        lansims = {}
        for agent1, agent2 in combinations(languages, 2):
            pair_name = f"{agent1} - {agent2}"
            lansim = language_similarity(
                language1=languages[agent1],
                language2=languages[agent2],
                length1=lengths[agent1],
                length2=lengths[agent2],
            )
            lansims[pair_name] = lansim

        lansims["mean"] = sum(lansims.values()) / len(lansims)
        return {
            "topsim": topsims,
            "lansim": lansims,
        }

    language_evaluator = LanguageEvaluator(
        agents=agents,
        input=dataset,
        metric=language_metric,
        logger=loggers,
        name="language",
    )
    tasks.append(language_evaluator)

    runner = TaskRunner(tasks)
    runner.run(n_iterations=cfg.n_iterations)

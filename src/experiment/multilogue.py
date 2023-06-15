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
from ..core.runner import Runner
from ..dataset import concept_dataset, random_split
from ..game import (
    MessageSignalingGame,
    MessageSignalingGameEvaluator,
    MessageSignalingGameResult,
    MessageSignalingGameTrainer,
)
from ..logger import WandBLogger
from ..loss import ConceptLoss
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
    n_agents: int

    # signal option
    sender_output: bool = False
    receiver_parrot: bool = False


def run_multilogue(config: dict):
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
        f"A{i}": ConceptOrMessageAgent(
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
        for i in range(cfg.n_agents)
    }

    optimizer_types = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizers = {
        agent_name: optimizer_types[cfg.optimizer](agent.parameters(), lr=cfg.lr)
        for agent_name, agent in agents.items()
    }

    agent_saver = ModelSaver(
        models=agents,
        path=f"{log_dir}/models",
    )
    agent_initializer = ModelInitializer(
        model=agents.values(),
    )

    loggers = [
        WandBLogger(project=cfg.wandb_project, name=cfg.wandb_name),
    ]

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
        sender_output=cfg.sender_output,
        receiver_parrot=cfg.receiver_parrot,
    )

    def game_metric(
        result: MessageSignalingGameResult,
    ):
        output_r = result.receiver_output.argmax(dim=-1)
        acc_part_r, acc_comp_r, acc_r = concept_accuracy(output_r, result.target)
        metrics = {
            "log_prob_s": result.sender_log_prob.mean().item(),
            "entropy_s": result.sender_entropy.mean().item(),
            "length_s": result.sender_length.float().mean().item(),
            "uniques_s": language_uniques(result.sender_message)
            / result.sender_message.shape[0],
            "acc_part_r": acc_part_r,
            "acc_comp": acc_comp_r,
        }
        metrics |= {f"acc_attr{str(i)}_r": acc for i, acc in enumerate(list(acc_r))}

        if cfg.sender_output:
            output_s = result.sender_output.argmax(dim=-1)
            acc_part_s, acc_comp_s, acc_s = concept_accuracy(output_s, result.target)
            metrics |= {"acc_part_s": acc_part_s, "acc_comp_s": acc_comp_s}
            metrics |= {f"acc_attr{i}_s": acc for i, acc in enumerate(list(acc_s))}

        return metrics

    game_train_evaluator = MessageSignalingGameEvaluator(
        game=game,
        agents=agents,
        input=train_dataset,
        target=train_dataset,
        metric=game_metric,
        logger=loggers,
        name="train",
        sender_output=cfg.sender_output,
        receiver_parrot=cfg.receiver_parrot,
    )
    game_valid_evaluator = MessageSignalingGameEvaluator(
        game=game,
        agents=agents,
        input=valid_dataset,
        target=valid_dataset,
        metric=game_metric,
        logger=loggers,
        name="valid",
        sender_output=cfg.sender_output,
        receiver_parrot=cfg.receiver_parrot,
    )

    def language_metric(
        input: np.ndarray,
        languages: dict[str, np.ndarray],
        lengths: dict[str, np.ndarray],
    ) -> dict:
        topsims = {}
        print("Computing topographic similarities...", end="", flush=True)
        for agent_name, language in languages.items():
            topsim = concept_topographic_similarity(concept=input, language=language)
            topsims[agent_name] = topsim
        topsims["mean"] = sum(topsims.values()) / len(topsims)
        print("done", flush=True)

        lansims = {}
        print("Computing language similarities...", end="", flush=True)
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
        print("done", flush=True)
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

    callbacks = [
        IntervalScheduler(agent_saver, 1000),
        IntervalScheduler(agent_initializer, 100000),
        game_trainer,
        IntervalScheduler(game_train_evaluator, 10),
        IntervalScheduler(game_valid_evaluator, 10),
        IntervalScheduler(language_evaluator, 1000, 7999),
    ]
    callbacks.extend(loggers)

    runner = Runner(callbacks)
    runner.run(n_iterations=cfg.n_iterations)

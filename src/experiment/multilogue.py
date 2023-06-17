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
    SignalingGame,
    SignalingGameEvaluator,
    SignalingGameResult,
    SignalingGameTrainer,
)
from ..logger import WandBLogger
from ..loss import ConceptLoss, SequenceMessageLoss
from ..message import SequenceMessage
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


def run_multilogue(config: dict):
    # make config
    cfg = Config(**config)

    # check device
    assert cfg.device in ["cpu", "cuda"]
    assert cfg.device == "cpu" or th.cuda.is_available()

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
    dataset = concept_dataset(cfg.n_attributes, cfg.n_values, device=cfg.device)
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

    # make optimizer
    optimizer_types = {
        "adam": th.optim.Adam,
        "sgd": th.optim.SGD,
    }
    optimizers = {
        agent_name: optimizer_types[cfg.optimizer](agent.parameters(), lr=cfg.lr)
        for agent_name, agent in agents.items()
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
    concept_loss = ConceptLoss(n_attributes=cfg.n_attributes, n_values=cfg.n_values).to(
        cfg.device
    )

    # make message loss
    baselines = {"batch_mean": BatchMeanBaseline}
    loss_baseline = baselines[cfg.baseline]()
    length_baseline = baselines[cfg.baseline]()
    message_loss = SequenceMessageLoss(
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        baseline=loss_baseline,
        length_baseline=length_baseline,
    ).to(cfg.device)

    # make game trainer
    game_trainer = SignalingGameTrainer(
        output_loss=concept_loss,
        message_loss=message_loss,
        agents=agents,
        optimizers=optimizers,
        dataloader=train_dataloader,
    )

    def game_metric(
        result: SignalingGameResult,
    ):
        output_r = result.receiver_output.argmax(dim=-1)
        acc_part_r, acc_comp_r, acc_r = concept_accuracy(output_r, result.target)
        message_s: SequenceMessage = result.sender_message
        metrics = {
            "log_prob_s": message_s.log_prob.mean().item(),
            "entropy_s": message_s.entropy.mean().item(),
            "length_s": message_s.length.float().mean().item(),
            "uniques_s": language_uniques(message_s.sequence) / message_s.batch_size,
            "acc_part_r": acc_part_r,
            "acc_comp": acc_comp_r,
        }
        metrics |= {f"acc_attr{str(i)}_r": acc for i, acc in enumerate(list(acc_r))}

        return metrics

    # make game evaluators
    game_evaluators = [
        SignalingGameEvaluator(
            agents=agents,
            input=dataset,
            target=dataset,
            metric=game_metric,
            logger=loggers,
            name=name,
        )
        for dataset, name in [(train_dataset, "train"), (valid_dataset, "valid")]
    ]

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

    # make language evaluator
    language_evaluator = LanguageEvaluator(
        agents=agents,
        input=dataset,
        metric=language_metric,
        logger=loggers,
        name="language",
    )

    # make callbacks
    callbacks = [
        IntervalScheduler(agent_saver, 1000),
        IntervalScheduler(agent_initializer, 100000),
        game_trainer,
        IntervalScheduler(game_evaluators, 10),
        # IntervalScheduler(language_evaluator, 1000, 7999),
    ]
    callbacks.extend(loggers)

    # make runner
    runner = Runner(callbacks)

    # run
    runner.run(n_iterations=cfg.n_iterations)

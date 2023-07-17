import datetime
import json
import os
import uuid
from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from ...analysis import language_similarity, topographic_similarity
from ...baseline import BatchMeanBaseline
from ...core import Evaluator, Interval, Runner, Trainer, fix_seed, init_weights
from ...dataset import random_split
from ...game import SignalingGame, SignalingGameResult
from ...logger import EarlyStopper, StepCounter, WandBLogger
from ...loss import DiscSeqReinforceLoss
from ...module import (
    DiscSeqMLPDecoder,
    DiscSeqMLPEncoder,
    DiscSeqRNNDecoder,
    DiscSeqRNNEncoder,
)


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
    input_length: int
    input_n_values: int
    message_length: int
    message_n_values: int
    entropy_weight: float
    length_weight: float
    baseline: str
    length_baseline: str
    run_sender_output: bool = False
    run_receiver_send: bool = False


class Agent(nn.Module):
    def __init__(
        self,
        input_encoder,
        input_decoder,
        message_encoder,
        message_decoder,
    ):
        super().__init__()
        self.input_encoder = input_encoder
        self.input_decoder = input_decoder
        self.message_encoder = message_encoder
        self.message_decoder = message_decoder

    def forward(self, command: str, input=None, message=None, latent=None):
        match command:
            case "input":
                return self.input_encoder(input)
            case "output":
                return self.input_decoder(latent)
            case "receive":
                return self.message_encoder(message)
            case "send":
                return self.message_decoder(latent)
            case "echo":
                return self.message_decoder(latent, message)
            case _:
                raise ValueError(f"Unknown command: {command}")


class DiscSeqAdapter(nn.Module):
    def __init__(self, module: nn.Module, padding: bool = False, eos: int = 0):
        super().__init__()
        self.module = module
        self.padding = padding
        self.eos = eos

    def forward(self, *args, **kwargs):
        x, logits = self.module(*args, **kwargs)

        distr = Categorical(logits=logits)
        log_prob = distr.log_prob(x)
        entropy = distr.entropy()

        mask = None
        length = None

        if self.padding:
            mask = x == self.eos
            indices = torch.argmax(mask.int(), dim=1)
            no_mask = ~mask.any(dim=1)
            indices[no_mask] = x.shape[1]
            mask = torch.arange(x.shape[1]).expand(x.shape).to(x.device)
            mask = (mask <= indices.unsqueeze(-1)).long()

            length = mask.sum(dim=-1)
            x = x * mask
            log_prob = log_prob * mask
            entropy = entropy * mask

        return x, {
            "log_prob": log_prob,
            "entropy": entropy,
            "length": length,
            "mask": mask,
            "logits": logits,
        }


def main(agents: dict[str, nn.Module], config: dict):
    # pre process
    cfg = Config(**config)

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
    for agent in agents.values():
        agent.apply(init_weights)

    optimizers = {
        name: optim.Adam(agent.parameters(), lr=cfg.lr)
        for name, agent in agents.items()
    }

    # data
    dataset = (
        torch.Tensor(
            list(product(torch.arange(cfg.input_n_values), repeat=cfg.input_length))
        )
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=cfg.batch_size, shuffle=False
    )

    # game
    baselines = {"batch_mean": BatchMeanBaseline}
    disc_seq_rf_loss = DiscSeqReinforceLoss(
        entropy_weight=cfg.entropy_weight,
        length_weight=cfg.length_weight,
        baseline=baselines[cfg.baseline](),
        length_baseline=baselines[cfg.length_baseline](),
    )

    def loss(result: SignalingGameResult):
        total_loss = 0
        # cross entropy loss for receiver output
        for logits_out_r in result.output_infos_r:
            logits_out_r = logits_out_r.view(-1, cfg.input_n_values)
            loss_out_r = F.cross_entropy(
                logits_out_r, result.input.view(-1), reduction="none"
            )
            loss_out_r = loss_out_r.view(-1, cfg.input_length).sum(dim=-1)
            total_loss += loss_out_r

        # reinforce loss for sender message
        log_prob_msg_s = result.message_info_s["log_prob"]
        entropy_msg_s = result.message_info_s["entropy"]
        length_msg_s = result.message_info_s["length"]
        loss_msg_s = disc_seq_rf_loss(
            reward=-total_loss.detach(),
            log_prob=log_prob_msg_s,
            entropy=entropy_msg_s,
            length=length_msg_s,
        )
        total_loss += loss_msg_s

        if result.output_s is not None:
            # cross entropy loss for sender output
            logits_out_s = result.output_info_s
            logits_out_s = logits_out_s.view(-1, cfg.input_n_values)
            loss_out_s = F.cross_entropy(
                logits_out_s, result.input.view(-1), reduction="none"
            )
            loss_out_s = loss_out_s.view(-1, cfg.input_length).sum(dim=-1)
            total_loss += loss_out_s

        if result.messages_r is not None:
            for message_info_r in result.message_infos_r:
                logits_msg_r = message_info_r["logits"]
                # cross entropy loss for receiver message
                logits_msg_r = logits_msg_r.view(-1, cfg.message_n_values)
                loss_msg_r = F.cross_entropy(
                    logits_msg_r, result.message_s.view(-1), reduction="none"
                )
                loss_msg_r = loss_msg_r.view(-1, cfg.message_length).sum(dim=-1)
                total_loss += loss_msg_r

        return total_loss

    class Metrics:
        def __init__(self, name: str, sender: str, receiver: str):
            self._name = name
            self._sender = sender
            self._receiver = receiver

        def __call__(self, results: list[SignalingGameResult]):
            result = results[0]
            mark = result.outputs_r[0] == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            metrics = {"acc_comp": acc_comp, "acc_part": acc_part}
            metrics |= {f"acc{i}": a for i, a in enumerate(list(acc))}
            metrics = {
                f"{self._name}.{self._sender}-{self._receiver}.{k}": v
                for k, v in metrics.items()
            }
            return metrics

    wandb_logger = WandBLogger(project=cfg.wandb_project)

    trainers = []
    evaluators = []
    for name_s, name_r in permutations(agents, 2):
        agent_s = agents[name_s]
        agent_r = agents[name_r]
        optimizer_s = optimizers[name_s]
        optimizer_r = optimizers[name_r]
        game = SignalingGame(
            agent_s,
            [agent_r],
            run_sender_output=cfg.run_sender_output,
            run_receiver_send=cfg.run_receiver_send,
        )
        trainer = Trainer(game, [optimizer_s, optimizer_r], train_dataloader, loss)
        trainers.append(trainer)
        train_evaluator = Evaluator(
            game, train_dataloader, Metrics("train", name_s, name_r), wandb_logger
        )
        valid_evaluator = Evaluator(
            game, valid_dataloader, Metrics("valid", name_s, name_r), wandb_logger
        )
        evaluators.extend([train_evaluator, valid_evaluator])

    # evaluation
    # early_stopper = EarlyStopper("valid.A0 -> A1.acc_comp", threshold=1 - 1e-6)

    # def drop_padding(x: np.ndarray):
    #     i = np.argwhere(x == 0)
    #     return x if len(i) == 0 else x[: i[0, 0]]

    # def lansim_metric(result: CollectiveInferringGameResult):
    #     names = list(result.agents)
    #     met = {}
    #     for name1, name2 in combinations(names, 2):
    #         output1: torch.Tensor = result.outputs[name1]
    #         output2: torch.Tensor = result.outputs[name2]
    #         output1 = output1.cpu().numpy()
    #         output2 = output2.cpu().numpy()

    #         lansim = language_similarity(
    #             output1,
    #             output2,
    #             dist="Levenshtein",
    #             processor=drop_padding,
    #             normalized=True,
    #         )
    #         met[f"{name1}-{name2}"] = lansim

    #     met["mean"] = np.mean(list(met.values()))
    #     return met

    # def topsim_metric(result: CollectiveInferringGameResult):
    #     input = result.input.cpu().numpy()
    #     met = {}
    #     for name in result.agents:
    #         output = result.outputs[name].cpu().numpy()
    #         topsim = topographic_similarity(
    #             input, output, y_processor=drop_padding, workers=-1
    #         )
    #         met[name] = topsim

    #     met["mean"] = np.mean(list(met.values()))
    #     return met

    # lansim_game = CollectiveInferringGame(output_command="send")
    # lansim_evals = [
    #     CollectiveInferringGameEvaluator(
    #         game=lansim_game,
    #         agents=agents,
    #         input=input,
    #         metric=lansim_metric,
    #         logger=wandb_logger,
    #         name=name,
    #     )
    #     for name, input in [
    #         ("train_lansim", train_dataset),
    #         ("valid_lansim", valid_dataset),
    #     ]
    # ]
    # topsim_evals = [
    #     CollectiveInferringGameEvaluator(
    #         game=lansim_game,
    #         agents=agents,
    #         input=input,
    #         metric=topsim_metric,
    #         logger=wandb_logger,
    #         name=name,
    #     )
    #     for name, input in [
    #         ("train_topsim", train_dataset),
    #         ("valid_topsim", valid_dataset),
    #     ]
    # ]

    runner = Runner(
        *trainers,
        *evaluators,
        StepCounter(wandb_logger),
        wandb_logger,
        # early_stop=early_stopper,
        use_tqdm=cfg.use_tqdm,
    )
    runner.run(cfg.n_epochs)

from statistics import fmean
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torchtyping import TensorType

from ...analysis import language_similarity, topographic_similarity
from ...core import Callback
from ...logger import Logger
from .agent import Agent
from .game import Game, GameResult


class GameMetrics(Callback):
    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        input: Iterable[Any],
        losses: dict[str, nn.Module],
        topsim_interval: int,
        heatmap_interval: int,
        metrics_loggers: Iterable[Logger],
        acc_comp_heatmap_loggers: Iterable[Logger],
        acc_part_heatmap_loggers: Iterable[Logger],
        game_option: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._agents = agents
        self._input = input
        self._losses = losses
        self._topsim_interval = topsim_interval
        self._heatmap_interval = heatmap_interval
        self._metrics_loggers = metrics_loggers
        self._acc_comp_heatmap_loggers = acc_comp_heatmap_loggers
        self._acc_part_heatmap_loggers = acc_part_heatmap_loggers
        self._game_option = game_option or {}

        self._agents_values = list(self._agents.values())
        self._agents_keys = list(self._agents.keys())
        self._games = [
            Game(sender, self._agents_values, **self._game_option)
            for sender in self._agents_values
        ]

        self._count = 0

    def on_update(self, step: int) -> None:
        # Switch to eval mode
        for agent in self._agents_values:
            agent.eval()

        # Execute the game for all agents and get the results
        input = next(iter(self._input))
        with torch.no_grad():
            results: list[GameResult] = [game(input) for game in self._games]

        # Calculate the accuracy heatmap for each receiver of the sender
        acc_comp_matrix = [
            [acc_comp(result.input, output_r) for output_r in result.output_r]
            for result in results
        ]
        acc_part_matrix = [
            [acc_part(result.input, output_r) for output_r in result.output_r]
            for result in results
        ]

        # Calculate the metrics
        acc_comp_mean = [fmean(mat) for mat in acc_comp_matrix]
        acc_comp_max = [max(mat) for mat in acc_comp_matrix]
        acc_comp_min = [min(mat) for mat in acc_comp_matrix]
        acc_part_mean = [fmean(mat) for mat in acc_part_matrix]
        acc_part_max = [max(mat) for mat in acc_part_matrix]
        acc_part_min = [min(mat) for mat in acc_part_matrix]

        uniques = [
            result.message_s.unique(dim=0).shape[0] / result.message_s.shape[0]
            for result in results
        ]
        length = [result.message_length_s.float().mean().item() for result in results]
        entropy = [result.message_entropy_s.mean().item() for result in results]
        losses = {
            k: [loss(result).mean() for result in results]
            for k, loss in self._losses.items()
        }

        metrics = {
            "acc_comp.mean.mean": fmean(acc_comp_mean),
            "acc_comp.max.max": max(acc_comp_mean),
            "acc_comp.min.min": min(acc_comp_mean),
            "acc_part.mean.mean": fmean(acc_part_mean),
            "acc_part.max.max": max(acc_part_mean),
            "acc_part.min.min": min(acc_part_mean),
            "unique.mean": fmean(uniques),
            "length.mean": fmean(length),
            "entropy.mean": fmean(entropy),
        }
        metrics |= {f"{k}_loss.mean": fmean(v) for k, v in losses.items()}

        for i, name in enumerate(self._agents_keys):
            metrics |= {
                f"acc_comp.mean.{name}": acc_comp_mean[i],
                f"acc_comp.max.{name}": acc_comp_max[i],
                f"acc_comp.min.{name}": acc_comp_min[i],
                f"acc_part.mean.{name}": acc_part_mean[i],
                f"acc_part.max.{name}": acc_part_max[i],
                f"acc_part.min.{name}": acc_part_min[i],
                f"unique.{name}": uniques[i],
                f"length.{name}": length[i],
                f"entropy.{name}": entropy[i],
            }
            metrics |= {f"{k}_loss.{name}": v[i] for k, v in losses.items()}

        if self._count % self._topsim_interval == 0:
            # Calculate the topographic similarity
            topsims = [
                topographic_similarity(
                    result.input.cpu().numpy(),
                    result.message_s.cpu().numpy(),
                    y_processor=drop_padding,  # type: ignore
                )
                for result in results
            ]
            metrics["topsim.mean"] = fmean(topsims)
            for i, name in enumerate(self._agents_keys):
                metrics[f"topsim.{name}"] = topsims[i]

        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

        for logger in self._metrics_loggers:
            logger.log(metrics)

        if self._count % self._heatmap_interval == 0:
            # Create heatmaps
            for matrix, loggers in [
                (acc_comp_matrix, self._acc_comp_heatmap_loggers),
                (acc_part_matrix, self._acc_part_heatmap_loggers),
            ]:
                df = pd.DataFrame(
                    matrix, columns=self._agents_keys, index=self._agents_keys
                )
                for logger in loggers:
                    logger.log(df)

        self._count += 1


class LanguageSimilarityMetrics(Callback):
    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        input: Iterable[Any],
        heatmap_interval: int,
        metrics_loggers: Iterable[Logger],
        heatmap_loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._agents = agents
        self._input = input
        self._heatmap_interval = heatmap_interval
        self._metrics_loggers = metrics_loggers
        self._heatmap_loggers = heatmap_loggers

        self._count = 0

    def on_update(self, step: int) -> None:
        self.calc()
        self._count += 1

    def calc(self) -> None:
        # Switch to eval mode
        for agent in self._agents.values():
            agent.eval()

        # Calculate the messages that each agent outputs for a single input
        input = next(iter(self._input))
        messages = []
        with torch.no_grad():
            for agent in self._agents.values():
                latent = agent(object=input, command="input")
                message, _ = agent(latent=latent, command="send")
                messages.append(message.cpu().numpy())

        # Calculate the language similarity between each pair of agents
        lansims = np.zeros((len(self._agents), len(self._agents)))
        for i in range(len(self._agents)):
            for j in range(i, len(self._agents)):
                lansim = language_similarity(
                    messages[i],
                    messages[j],
                    processor=drop_padding,
                )
                lansims[i, j] = lansim
                lansims[j, i] = lansim

        # Log the language similarity
        lansim_means = lansims.mean(axis=1)
        lansim_mean = lansim_means.mean()
        metrics = {
            "lansim.mean": lansim_mean,
        }

        # Log the language similarity for each agent
        metrics |= {
            f"lansim.mean.{name}": lansim_mean
            for name, lansim_mean in zip(self._agents, lansim_means)
        }

        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

        for logger in self._metrics_loggers:
            logger.log(metrics)

        # Save a heatmap of the language similarity
        if self._count % self._heatmap_interval == 0:
            df = pd.DataFrame(data=lansims, index=self._agents, columns=self._agents)
            for logger in self._heatmap_loggers:
                logger.log(df)


BATCH = "batch"
LENGTH = "length"


def acc_comp(
    input: TensorType[BATCH, LENGTH, int],
    output: TensorType[BATCH, LENGTH, int],
) -> float:
    return float((output == input).all(dim=-1).float().mean().item())


def acc_part(
    input: TensorType[BATCH, LENGTH, int],
    output: TensorType[BATCH, LENGTH, int],
) -> float:
    return float((output == input).float().mean().item())


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]

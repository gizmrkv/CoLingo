from statistics import fmean
from typing import Any, Iterable

import numpy as np
import pandas as pd
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
        topsim_interval: int,
        metrics_loggers: Iterable[Logger],
        acc_comp_heatmap_loggers: Iterable[Logger],
        acc_part_heatmap_loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._agents = agents
        self._input = input
        self._topsim_interval = topsim_interval
        self._metrics_loggers = metrics_loggers
        self._acc_comp_heatmap_loggers = acc_comp_heatmap_loggers
        self._acc_part_heatmap_loggers = acc_part_heatmap_loggers

        self._agents_values = list(self._agents.values())
        self._agents_keys = list(self._agents.keys())
        self._games = [
            Game(sender, self._agents_values) for sender in self._agents_values
        ]

    def on_update(self, step: int) -> None:
        # Execute the game for all agents and get the results
        input = next(iter(self._input))
        results: list[GameResult] = [game(input) for game in self._games]

        # Create a heatmap of the accuracy rate for all agent pairs
        acc_comp_matrix = [
            [acc_comp(result.input, output_r) for output_r in result.output_r]
            for result in results
        ]
        acc_part_matrix = [
            [acc_part(result.input, output_r) for output_r in result.output_r]
            for result in results
        ]

        # Calculate the metrics
        acc_comp_means = [fmean(acc_comp_matrix[i]) for i in range(len(self._agents))]
        acc_part_means = [fmean(acc_part_matrix[i]) for i in range(len(self._agents))]
        uniques = [
            result.message_s.unique(dim=0).shape[0] / result.message_s.shape[0]
            for result in results
        ]
        length = [result.message_length_s.float().mean().item() for result in results]
        entropy = [result.message_entropy_s.mean().item() for result in results]

        metrics = {
            "acc_comp.mean": fmean(acc_comp_means),
            "acc_part.mean": fmean(acc_part_means),
            "unique": fmean(uniques),
            "length": fmean(length),
            "entropy": fmean(entropy),
        }

        for i, name in enumerate(self._agents_keys):
            metrics[f"{name}.acc_comp.mean"] = acc_comp_means[i]
            metrics[f"{name}.acc_part.mean"] = acc_part_means[i]
            metrics[f"{name}.unique"] = uniques[i]
            metrics[f"{name}.length"] = length[i]
            metrics[f"{name}.entropy"] = entropy[i]

        if step % self._topsim_interval == 0:
            # Calculate the topographic similarity
            topsims = [
                topographic_similarity(
                    result.input.cpu().numpy(),
                    result.message_s.cpu().numpy(),
                    y_processor=drop_padding,  # type: ignore
                )
                for result in results
            ]
            metrics["topsim_mean"] = fmean(topsims)
            for i, name in enumerate(self._agents_keys):
                metrics[f"{name}.topsim"] = topsims[i]

        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

        for logger in self._metrics_loggers:
            logger.log(metrics)

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


class LanguageSimilarityMetrics(Callback):
    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        input: Iterable[Any],
        metrics_loggers: Iterable[Logger],
        heatmap_loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._agents = agents
        self._input = input
        self._metrics_loggers = metrics_loggers
        self._heatmap_loggers = heatmap_loggers

    def on_update(self, step: int) -> None:
        self.calc()

    def calc(self) -> None:
        # Calculate the messages that each agent outputs for a single input
        input = next(iter(self._input))
        messages = []
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
            "lansim_mean": lansim_mean,
        }

        # Log the language similarity for each agent
        metrics |= {
            f"{name}.lansim_mean": lansim_mean
            for name, lansim_mean in zip(self._agents, lansim_means)
        }

        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

        for logger in self._metrics_loggers:
            logger.log(metrics)

        # Save a heatmap of the language similarity
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

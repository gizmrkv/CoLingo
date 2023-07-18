from statistics import mean
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from torch import nn

import wandb

from ...analysis import language_similarity, topographic_similarity
from ...core import Callback
from ...logger import Logger
from .agent import Agent
from .game import Game, GameResult


class Metrics(Logger):
    def __init__(
        self,
        name: str,
        sender_name: str,
        receiver_names: Sequence[str],
        loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._sender_name = sender_name
        self._receiver_names = receiver_names
        self._loggers = loggers

    def log(self, results: list[GameResult | tuple[GameResult, float]]) -> None:
        metrics = []
        for result in results:
            if isinstance(result, tuple):
                metrics.append(self.calc_metrics(*result))
            else:
                metrics.append(self.calc_metrics(result))
        mean_metrics = {k: mean(m[k] for m in metrics) for k in metrics[0]}
        mean_metrics = {
            f"{self._name}.{self._sender_name}->{k}": v for k, v in mean_metrics.items()
        }
        for logger in self._loggers:
            logger.log(mean_metrics)

    def calc_metrics(
        self, result: GameResult, loss: float | None = None
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = loss

        for name_r, output_r in zip(self._receiver_names, result.output_r):
            mark = output_r == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            metrics |= {
                f"{name_r}.acc_comp": acc_comp,
                f"{name_r}.acc_part": acc_part,
            }
            metrics |= {f"{name_r}.acc{i}": a.item() for i, a in enumerate(list(acc))}

        return metrics


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


class TopographicSimilarity(Callback):
    def __init__(
        self,
        name: str,
        sender: nn.Module,
        dataloader: Iterable[Any],
        loggers: Iterable[Logger],
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ) -> None:
        self._name = name
        self._sender = sender
        self._dataloader = dataloader
        self._loggers = loggers
        self._run_on_begin = run_on_begin
        self._run_on_end = run_on_end

    def on_begin(self) -> None:
        if self._run_on_begin:
            self.evaluate()

    def on_update(self, step: int) -> None:
        self.evaluate()

    def on_end(self) -> None:
        if self._run_on_end:
            self.evaluate()

    def evaluate(self) -> None:
        topsims = []
        for input in self._dataloader:
            latent = self._sender(object=input, command="input")
            message, _ = self._sender(latent=latent, command="send")
            topsim = topographic_similarity(
                input.cpu().numpy(), message.cpu().numpy(), y_processor=drop_padding  # type: ignore
            )
            topsims.append(topsim)

        mean_topsim = mean(topsims)

        for logger in self._loggers:
            logger.log({f"{self._name}.topsim": mean_topsim})


class LanguageSimilarity(Callback):
    def __init__(
        self,
        name: str,
        sender1: nn.Module,
        sender2: nn.Module,
        dataloader: Iterable[Any],
        loggers: Iterable[Logger],
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ) -> None:
        self._name = name
        self._sender1 = sender1
        self._sender2 = sender2
        self._dataloader = dataloader
        self._loggers = loggers
        self._run_on_begin = run_on_begin
        self._run_on_end = run_on_end

    def on_begin(self) -> None:
        if self._run_on_begin:
            self.evaluate()

    def on_update(self, step: int) -> None:
        self.evaluate()

    def on_end(self) -> None:
        if self._run_on_end:
            self.evaluate()

    def evaluate(self) -> None:
        lansims = []
        for input in self._dataloader:
            latent1 = self._sender1(object=input, command="input")
            message1, _ = self._sender1(latent=latent1, command="send")
            latent2 = self._sender2(object=input, command="input")
            message2, _ = self._sender2(latent=latent2, command="send")
            lansim = language_similarity(
                message1.cpu().numpy(), message2.cpu().numpy(), processor=drop_padding
            )
            lansims.append(lansim)

        mean_lansim = mean(lansims)

        for logger in self._loggers:
            logger.log({f"{self._name}.lansim": mean_lansim})


class AccuracyMatrix(Callback):
    def __init__(
        self,
        length: int,
        name: str,
        agents: dict[str, Agent],
        dataloader: Iterable[Any],
    ) -> None:
        self._length = length
        self._name = name
        self._agents = agents
        self._dataloader = dataloader

        self._games = [
            Game(sender, list(agents.values())) for name, sender in agents.items()
        ]

    def on_end(self) -> None:
        self.evaluate()

    def evaluate(self) -> None:
        comp_heatmap = []
        part_heatmap = []
        attr_heatmap: list[list[list[float]]] = [[] for i in range(self._length)]

        input = next(iter(self._dataloader))
        for game in self._games:
            result: GameResult = game(input)
            comp, part, attr = self.calc_acc_comps(result)
            comp_heatmap.append(comp)
            part_heatmap.append(part)
            for i, a in enumerate(attr):
                attr_heatmap[i].append(a)

        index = list(self._agents.keys())

        comp_df = pd.DataFrame(data=comp_heatmap, index=index, columns=index)
        plt.figure()
        sns.heatmap(comp_df)
        wandb.log({f"{self._name}.acc_comp": wandb.Image(plt)})

        part_df = pd.DataFrame(data=part_heatmap, index=index, columns=index)
        plt.figure()
        sns.heatmap(part_df)
        wandb.log({f"{self._name}.acc_part": wandb.Image(plt)})

        for i, attr in enumerate(attr_heatmap):
            attr_df = pd.DataFrame(data=attr, index=index, columns=index)
            plt.figure()
            sns.heatmap(attr_df)
            wandb.log({f"{self._name}.acc{i}": wandb.Image(plt)})

    def calc_acc_comps(
        self, result: GameResult
    ) -> tuple[list[float], list[float], list[list[float]]]:
        acc_comps = []
        acc_parts = []
        acc_attrs: list[list[float]] = [[] for i in range(self._length)]

        for output_r in result.output_r:
            mark = output_r == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            acc_comps.append(acc_comp)
            acc_parts.append(acc_part)
            for i, a in enumerate(list(acc)):
                acc_attrs[i].append(a.item())

        return acc_comps, acc_parts, acc_attrs

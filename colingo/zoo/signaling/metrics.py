from statistics import mean
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

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
        mean_metrics = {f"{self._name}.{k}": v for k, v in mean_metrics.items()}
        for logger in self._loggers:
            logger.log(mean_metrics)

    def calc_metrics(
        self, result: GameResult, loss: float | None = None
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = loss

        # acc
        for name_r, output_r in zip(self._receiver_names, result.output_r):
            mark = output_r == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            metrics |= {
                f"{self._sender_name}->{name_r}.acc_comp": acc_comp,
                f"{self._sender_name}->{name_r}.acc_part": acc_part,
            }
            # metrics |= {f"{self._sender_name}->{name_r}.acc{i}": a.item() for i, a in enumerate(list(acc))}

        # message
        n_unique = result.message_s.unique(dim=0).shape[0]
        metrics[f"{self._sender_name}.unique"] = n_unique / result.message_s.shape[0]
        metrics[f"{self._sender_name}.msg_ent"] = result.message_entropy_s.mean().item()
        metrics[f"{self._sender_name}.msg_len"] = (
            result.message_length_s.float().mean().item()
        )

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
        input = next(iter(self._dataloader))
        latent = self._sender(object=input, command="input")
        message, _ = self._sender(latent=latent, command="send")
        topsim = topographic_similarity(
            input.cpu().numpy(), message.cpu().numpy(), y_processor=drop_padding  # type: ignore
        )

        for logger in self._loggers:
            logger.log({f"{self._name}.topsim": topsim})


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
        input = next(iter(self._dataloader))
        latent1 = self._sender1(object=input, command="input")
        message1, _ = self._sender1(latent=latent1, command="send")
        latent2 = self._sender2(object=input, command="input")
        message2, _ = self._sender2(latent=latent2, command="send")
        lansim = language_similarity(
            message1.cpu().numpy(), message2.cpu().numpy(), processor=drop_padding
        )

        for logger in self._loggers:
            logger.log({f"{self._name}.lansim": lansim})


class AccuracyMatrix(Callback):
    def __init__(
        self,
        length: int,
        name: str,
        agents: dict[str, Agent],
        dataloader: Iterable[Any],
        loggers: Iterable[Logger],
    ) -> None:
        self._length = length
        self._name = name
        self._agents = agents
        self._dataloader = dataloader
        self._loggers = loggers

        self._games = [
            Game(sender, list(agents.values())) for name, sender in agents.items()
        ]
        self._index = list(self._agents.keys())

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

        images = {}

        images[f"{self._name}.acc_comp_mat"] = self.get_wandb_image(comp_heatmap)
        images[f"{self._name}.acc_part_mat"] = self.get_wandb_image(part_heatmap)

        for i, attr in enumerate(attr_heatmap):
            images[f"{self._name}.acc_attr{i}_mat"] = self.get_wandb_image(attr)

        for logger in self._loggers:
            logger.log(images)

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

    def get_wandb_image(self, data: list[list[float]]) -> wandb.Image:
        df = pd.DataFrame(data=data, index=self._index, columns=self._index)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="inferno")
        ax.set_ylabel("Sender")
        ax.set_xlabel("Receiver")
        image = wandb.Image(fig)
        plt.close()
        return image


class LanguageSimilarityMatrix(Callback):
    def __init__(
        self,
        name: str,
        agents: dict[str, Agent],
        dataloader: Iterable[Any],
        loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._agents = agents
        self._dataloader = dataloader
        self._loggers = loggers
        self._index = list(self._agents.keys())

    def on_end(self) -> None:
        self.evaluate()

    def evaluate(self) -> None:
        input = next(iter(self._dataloader))
        lansims = [[0.0] * len(self._agents) for _ in range(len(self._agents))]
        for i in range(len(self._agents)):
            for j in range(i, len(self._agents)):
                sender1 = self._agents[self._index[i]]
                sender2 = self._agents[self._index[j]]
                latent1 = sender1(object=input, command="input")
                message1, _ = sender1(latent=latent1, command="send")
                latent2 = sender2(object=input, command="input")
                message2, _ = sender2(latent=latent2, command="send")
                lansim = language_similarity(
                    message1.cpu().numpy(),
                    message2.cpu().numpy(),
                    processor=drop_padding,
                )
                lansims[i][j] = lansim
                lansims[j][i] = lansim

        df = pd.DataFrame(data=lansims, index=self._index, columns=self._index)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(df, vmin=0, vmax=1, annot=True, fmt=".2f", cmap="inferno")
        image = wandb.Image(fig)
        plt.close()

        for logger in self._loggers:
            logger.log({f"{self._name}.lansim_mat": image})

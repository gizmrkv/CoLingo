import os
from glob import glob
from statistics import mean
from typing import Any, Iterable, Sequence

import matplotlib
from moviepy.editor import ImageSequenceClip

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from torch import nn
from torchtyping import TensorType

import wandb

from ...analysis import language_similarity, topographic_similarity
from ...core import Callback
from ...logger import Logger
from .agent import Agent
from .game import Game, GameResult


class OldAccuracyMatrix(Callback):
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


class GameMetricsLogger(Logger):
    def __init__(
        self,
        name: str,
        sender_name: str,
        topsim_interval: int,
        loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._sender_name = sender_name
        self._topsim_interval = topsim_interval
        self._loggers = loggers
        self._step = 0

    def log(self, result: GameResult) -> None:
        # Calculate the average length and entropy of the message
        metrics = {
            "length_mean": result.message_length_s.float().mean().item(),
            "entropy_mean": result.message_entropy_s.mean().item(),
        }

        # Calculate the accuracy of the complete answer and the partial answer
        acc_comps = [acc_comp(result.input, output_r) for output_r in result.output_r]
        acc_parts = [acc_part(result.input, output_r) for output_r in result.output_r]
        metrics |= {"acc_comp_mean": mean(acc_comps), "acc_part_mean": mean(acc_parts)}

        # Calculate the proportion of unique messages
        n_uniques = result.message_s.unique(dim=0).shape[0]
        metrics["unique"] = n_uniques / result.message_s.shape[0]

        if self._step % self._topsim_interval == 0:
            # Calculate the topographic similarity
            metrics["topsim"] = topographic_similarity(
                result.input.cpu().numpy(), result.message_s.cpu().numpy(), y_processor=drop_padding  # type: ignore
            )

        metrics = {
            f"{self._name}.{self._sender_name}.{k}": v for k, v in metrics.items()
        }

        for logger in self._loggers:
            logger.log(metrics)

    def on_update(self, step: int) -> None:
        self._step = step


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


class LanguageSimilarityMetrics(Callback):
    def __init__(
        self,
        path: str,
        name: str,
        agents: dict[str, Agent],
        input: Iterable[Any],
        frame_interval: int,
        loggers: Iterable[Logger],
    ) -> None:
        self._path = path
        self._name = name
        self._agents = agents
        self._input = input
        self._interval = frame_interval
        self._loggers = loggers
        self._step = 0
        self._count = 0

        os.makedirs(f"{self._path}/lansim_frames", exist_ok=True)

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

        if self._step % self._interval == 0:
            # Save a heatmap of the language similarity
            df = pd.DataFrame(data=lansims, index=self._agents, columns=self._agents)
            sns.heatmap(
                df,
                vmin=0.0,
                vmax=1.0,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                square=True,
                cbar=True,
            )
            plt.title(f"lansim step: {self._step}")
            plt.savefig(f"{self._path}/lansim_frames/{self._count:0>6}.png")
            plt.clf()
            self._count += 1

        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}

        for logger in self._loggers:
            logger.log(metrics)

    def on_update(self, step: int) -> None:
        self._step = step
        self.calc()

    def on_end(self) -> None:
        # Save a video of the language similarity heatmap
        frames = sorted(glob(f"{self._path}/lansim_frames/*.png"))
        name = f"{self._path}/{self._name}_lansim.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(name)
        for logger in self._loggers:
            logger.log({f"{self._name}.lansim_heatmap": wandb.Video(name)})

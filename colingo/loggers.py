import math
import shutil
import time
from pathlib import Path
from statistics import fmean, stdev
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    TypeVar,
)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray

import wandb

from .analysis import language_similarity, topographic_similarity
from .core import Language, Loggable, Stoppable, Task

matplotlib.use("Agg")


class KeyPrefix(Loggable[Mapping[str, Any]]):
    def __init__(
        self,
        prefix: str,
        loggers: Iterable[Loggable[Mapping[str, Any]]],
    ) -> None:
        self.prefix = prefix
        self.loggers = loggers

    def log(self, input: Mapping[str, Any], step: int | None = None) -> None:
        input = {self.prefix + k: v for k, v in input.items()}
        for logger in self.loggers:
            logger.log(input)


class KeySuffix(Loggable[Mapping[str, Any]]):
    def __init__(
        self,
        suffix: str,
        loggers: Iterable[Loggable[Mapping[str, Any]]],
    ) -> None:
        self.suffix = suffix
        self.loggers = loggers

    def log(self, input: Mapping[str, Any], step: int | None = None) -> None:
        input = {k + self.suffix: v for k, v in input.items()}
        for logger in self.loggers:
            logger.log(input)


class WandbLogger(Task, Loggable[Mapping[str, Any]]):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self.inputs: Dict[str, float] = {}

    def log(self, input: Mapping[str, Any], step: int | None = None) -> None:
        self.inputs.update(input)

    def flush(self) -> None:
        wandb.log(self.inputs)
        self.inputs.clear()

    def on_begin(self) -> None:
        self.flush()

    def on_update(self, step: int) -> None:
        self.flush()

    def on_end(self) -> None:
        self.flush()
        wandb.finish()

    def priority(self) -> int:
        return 100


T = TypeVar("T")
U = TypeVar("U")


class LambdaLogger(Loggable[T], Generic[T, U]):
    def __init__(self, func: Callable[[T], U], loggers: Iterable[Loggable[U]]) -> None:
        self.func = func
        self.loggers = loggers

    def log(self, input: T, step: int | None = None) -> None:
        for logger in self.loggers:
            logger.log(self.func(input), step)


class HeatmapLogger(Loggable[NDArray[np.float32]]):
    def __init__(
        self,
        path: Path,
        cleanup: bool = False,
        heatmap_option: Mapping[str, Any] | None = None,
        loggers: Iterable[Loggable[Path]] | None = None,
    ) -> None:
        self.path = path
        self.cleanup = cleanup
        self.heatmap_option = heatmap_option or {}
        self.loggers = loggers or []

        path.mkdir(parents=True, exist_ok=True)

    def log(self, input: NDArray[np.float32], step: int | None = None) -> None:
        sns.heatmap(input, **self.heatmap_option)
        step = -1 if step is None else step
        plt.title(f"step: {step}")
        path = self.path.joinpath(f"{step:0>8}.png")
        plt.savefig(path)
        plt.clf()

        for logger in self.loggers:
            logger.log(path, step)


class ImageToVideoTask(Task):
    def __init__(
        self,
        src_dir: Path,
        tgt_dir: Path | None = None,
        cleanup: bool = False,
        loggers: Iterable[Loggable[Path]] | None = None,
    ) -> None:
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir or src_dir
        self.cleanup = cleanup
        self.loggers = loggers or []

        self.tgt_dir.mkdir(parents=True, exist_ok=True)

    def on_end(self) -> None:
        # Generates a video from the saved heatmap frames.
        frames_path = list(self.src_dir.glob("*.png"))
        frames = sorted([f.as_posix() for f in frames_path])
        path = self.tgt_dir.joinpath("video.mp4")
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(path.as_posix())

        for logger in self.loggers:
            logger.log(path)

        if self.cleanup:
            # Delete the frames
            shutil.rmtree(self.src_dir)

    def priority(self) -> int:
        return 10000


class DictStopper(Task, Loggable[Mapping[str, Any]], Stoppable):
    def __init__(self, pred: Callable[[Mapping[str, Any]], bool]) -> None:
        self.pred = pred
        self.inputs: Dict[str, float] = {}
        self._stop = False

    def log(self, input: Mapping[str, Any], step: int | None = None) -> None:
        self.inputs.update(input)

    def stop(self, step: int) -> bool:
        return self._stop

    def on_update(self, step: int) -> None:
        self._stop = self.pred(self.inputs)
        self.inputs.clear()

    def on_begin(self) -> None:
        self.inputs.clear()

    def on_end(self) -> None:
        self.inputs.clear()

    def priority(self) -> int:
        return 100


class KeyChecker(Task, Loggable[Mapping[str, Any]]):
    def __init__(self) -> None:
        self.seen: set[str] = set()

    def log(self, input: Mapping[str, Any], step: int | None = None) -> None:
        for key in input:
            if key in self.seen:
                raise ValueError(f"Duplicate key: {key}")
            self.seen.add(key)

    def on_begin(self) -> None:
        self.seen.clear()

    def on_update(self, step: int) -> None:
        self.seen.clear()

    def on_end(self) -> None:
        self.seen.clear()

    def priority(self) -> int:
        return 100


class StepCounter(Task):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, int]]]) -> None:
        self.loggers = loggers

    def on_update(self, step: int) -> None:
        for logger in self.loggers:
            logger.log({"step": step})

    def priority(self) -> int:
        return 10000


class Stopwatch(Task):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, float]]]) -> None:
        super().__init__()
        self.loggers = loggers
        self.start_time = time.time()

    def on_update(self, step: int) -> None:
        elapsed_time = time.time() - self.start_time
        for logger in self.loggers:
            logger.log({"elapsed_time": elapsed_time})

    def priority(self) -> int:
        return 10000


class TimeDebugger(Task):
    def __init__(self, tasks: Iterable[Task]) -> None:
        self.tasks = tasks

    def on_update(self, step: int) -> None:
        for i, task in enumerate(self.tasks):
            torch.cuda.synchronize()
            start = time.time()
            task.on_update(step)
            torch.cuda.synchronize()
            end = time.time()
            print(f"{i}st time: {end - start:.3f} sec")

    def priority(self) -> int:
        return 10000


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


L = TypeVar("L", bound=Language)


class TopographicSimilarityLogger(Loggable[L], Generic[L]):
    def __init__(self, loggers: Iterable[Loggable[Mapping[str, float]]]) -> None:
        self.loggers = loggers

    def log(self, lang: L, step: int | None = None) -> None:
        concept = lang.concept()
        messages = lang.messages()
        topsims = {
            f"topsim.{k}": topographic_similarity(
                concept.cpu().numpy(),
                msg.cpu().numpy(),
                y_processor=drop_padding,  # type: ignore
            )
            for k, msg in messages.items()
        }

        topsims = {k: (0.0 if math.isnan(v) else v) for k, v in topsims.items()}

        topsims |= {
            f"topsim.mean": fmean(topsims.values()),
            f"topsim.std": stdev(topsims.values()),
            f"topsim.max": max(topsims.values()),
            f"topsim.min": min(topsims.values()),
        }

        for logger in self.loggers:
            logger.log(topsims)


class LanguageSimilarityLogger(Loggable[L], Generic[L]):
    def __init__(
        self,
        order: Collection[str],
        loggers: Iterable[Loggable[Mapping[str, float]]],
        heatmap_logger: HeatmapLogger | None = None,
    ) -> None:
        self.order = order
        self.loggers = loggers
        self.heatmap_logger = heatmap_logger

    def log(self, lang: L, step: int | None = None) -> None:
        messages = lang.messages()
        msgs = []
        for k in self.order:
            msgs.append(messages[k].cpu().numpy())

        matrix: List[List[float]] = []
        for _ in msgs:
            matrix.append([0.0] * len(msgs))

        for i in range(len(self.order)):
            for j in range(i, len(self.order)):
                ls = language_similarity(msgs[i], msgs[j], processor=drop_padding)
                matrix[i][j] = ls
                matrix[j][i] = ls

        if self.heatmap_logger is not None:
            df = pd.DataFrame(matrix, columns=self.order, index=self.order)
            self.heatmap_logger.log(df, step)

        langsims = {}
        for i, (k, langsim) in enumerate(zip(self.order, matrix)):
            langsim = langsim[:i] + langsim[i + 1 :]
            langsims[f"{k}.langsim.mean"] = fmean(langsim)
            langsims[f"{k}.langsim.std"] = stdev(langsim)
            langsims[f"{k}.langsim.max"] = max(langsim)
            langsims[f"{k}.langsim.min"] = min(langsim)

        for logger in self.loggers:
            logger.log(langsims)


class LanguageLogger(Loggable[L], Generic[L]):
    def __init__(
        self,
        path: Path,
        loggers: Iterable[Loggable[Path]] | None = None,
    ) -> None:
        self.path = path
        self.loggers = loggers or []

        path.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        input: L,
        step: int | None = None,
    ) -> None:
        concept = input.concept()
        messages = input.messages()
        lines = [
            ", ".join(
                [f"c{i}" for i in range(concept.shape[1])] + list(messages.keys())
            )
        ]
        for i in range(concept.shape[0]):
            cpt = concept[i]
            line = ", ".join([str(c) for c in cpt.tolist()])
            for message in messages.values():
                msg = message[i]
                j = torch.argwhere(msg == 0)
                msg = msg if len(j) == 0 else msg[: j[0, 0]]
                line += ", " + "-".join([str(c) for c in msg.tolist()])

            lines.append(line)

        lang = "\n".join(lines)
        path = self.path.joinpath(f"{step:0>8}.csv")

        with path.open("w") as f:
            f.write(lang)

        for logger in self.loggers:
            logger.log(path, step)

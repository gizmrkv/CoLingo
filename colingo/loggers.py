import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray
from torchtyping import TensorType

import wandb

from .core import Loggable, Task

matplotlib.use("Agg")


class Namer(Loggable[Mapping[str, Any]]):
    def __init__(
        self,
        name: str,
        loggers: Iterable[Loggable[Mapping[str, Any]]],
        joiner: str = ".",
    ) -> None:
        self.name = name
        self.loggers = loggers
        self.joiner = joiner

    def log(self, input: Mapping[str, Any]) -> None:
        input = {f"{self.name}{self.joiner}{k}": v for k, v in input.items()}
        for logger in self.loggers:
            logger.log(input)


class WandbLogger(Task, Loggable[Mapping[str, Any]]):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self.inputs: Dict[str, float] = {}

    def log(self, input: Mapping[str, Any]) -> None:
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


class HeatmapLogger(Task, Loggable[Tuple[int, NDArray[np.float32]]]):
    def __init__(
        self,
        save_dir: Path,
        cleanup: bool = False,
        heatmap_option: Mapping[str, Any] | None = None,
        loggers: Iterable[Loggable[Path]] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.cleanup = cleanup
        self.heatmap_option = heatmap_option or {}
        self.loggers = loggers or []

        save_dir.mkdir(parents=True, exist_ok=True)

    def log(self, input: Tuple[int, NDArray[np.float32]]) -> None:
        step, data = input
        sns.heatmap(data, **self.heatmap_option)
        plt.title(f"step: {step}")
        plt.savefig(self.save_dir.joinpath(f"{step:0>8}.png"))
        plt.clf()

    def on_end(self) -> None:
        # Generates a video from the saved heatmap frames.
        frames_path = list(self.save_dir.glob("*.png"))
        frames = sorted([f.as_posix() for f in frames_path])
        path = self.save_dir.joinpath("video.mp4").as_posix()
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(path)

        for logger in self.loggers:
            logger.log(Path(path))

        if self.cleanup:
            # Delete the frames
            shutil.rmtree(self.save_dir)


class LanguageLogger(Loggable[Tuple[int, TensorType[..., int], TensorType[..., int]]]):
    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    def log(
        self, input: Tuple[int, TensorType[..., int], TensorType[..., int]]
    ) -> None:
        step, concept, message = input
        lines = []
        for cpt, msg in zip(concept, message):
            i = torch.argwhere(msg == 0)
            msg = msg if len(i) == 0 else msg[: i[0, 0]]

            c = ",".join([str(i) for i in cpt.tolist()])
            m = "-".join([str(i) for i in msg.tolist()])

            lines.append(",".join([c, m]) + "\n")

        lang = "".join(lines)

        with self.save_dir.joinpath(f"{step}.csv").open("w") as f:
            f.write(lang)

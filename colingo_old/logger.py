import os
import shutil
from glob import glob
from typing import Any, Callable, Dict, Iterable, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray

import wandb

from .core import Callback

matplotlib.use("Agg")


class Logger(Callback):
    def log(self, metrics: Any) -> None:
        pass


class WandBLogger(Logger):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self._metrics: Dict[str, float] = {}

    def log(self, metrics: Mapping[str, float]) -> None:
        self._metrics.update(metrics)

    def flush(self) -> None:
        if len(self._metrics) > 0:
            wandb.log(self._metrics)
        self._metrics.clear()

    def on_begin(self) -> None:
        self.flush()

    def on_update(self, step: int) -> None:
        self.flush()

    def on_end(self) -> None:
        self.flush()


class DuplicateChecker(Logger):
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def log(self, metrics: Iterable[str]) -> None:
        for k in metrics:
            if k in self._seen:
                raise ValueError(f"Duplicate key: {k}")
            self._seen.add(k)

    def on_begin(self) -> None:
        self._seen.clear()

    def on_update(self, step: int) -> None:
        self._seen.clear()

    def on_end(self) -> None:
        self._seen.clear()


class EarlyStopper(Logger):
    def __init__(self, pred: Callable[[Any], bool]) -> None:
        self._pred = pred
        self._stop = False

    def log(self, metrics: Any) -> None:
        self._stop = self._pred(metrics)

    def __call__(self, step: int) -> bool:
        return self._stop


class HeatmapLogger(Logger):
    def __init__(
        self,
        save_dir: str,
        name: str,
        wandb_loggers: Iterable[Logger],
        write_video: bool = True,
        delete_frames: bool = True,
        heatmap_option: Mapping[str, Any] | None = None,
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        self._wandb_loggers = wandb_loggers
        self._write_video = write_video
        self._delete_frames = delete_frames
        self._heatmap_option = heatmap_option or {}
        self._step = 0
        self._frames_dir = f"{self._save_dir}/{self._name}_frames"

        os.makedirs(self._frames_dir, exist_ok=True)

    def log(self, data: NDArray[np.float32]) -> None:
        # Save a heatmap frame
        sns.heatmap(data, **self._heatmap_option)
        plt.title(f"{self._name} step: {self._step}")
        plt.savefig(f"{self._frames_dir}/{self._step:0>8}.png")
        plt.clf()

    def on_update(self, step: int) -> None:
        self._step = step

    def on_end(self) -> None:
        if self._write_video:
            # Save a video of the heatmap
            frames = sorted(glob(f"{self._frames_dir}/*.png"))
            name = f"{self._save_dir}/{self._name}.mp4"
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(name)

            for logger in self._wandb_loggers:
                logger.log({f"{self._name}": wandb.Video(name)})

        if self._delete_frames:
            # Delete the frames
            shutil.rmtree(self._frames_dir)

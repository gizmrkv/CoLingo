import os
import shutil
from glob import glob
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray

import wandb

from .core import RunnerCallback

matplotlib.use("Agg")


class WandbLogger(RunnerCallback):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self.metrics: dict[str, float] = {}

    def __call__(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def flush(self) -> None:
        wandb.log(self.metrics)
        self.metrics.clear()

    def on_begin(self) -> None:
        self.flush()

    def on_update(self, step: int) -> None:
        self.flush()

    def on_end(self) -> None:
        self.flush()


class HeatmapLogger(RunnerCallback):
    def __init__(
        self,
        save_dir: str,
        name: str,
        wandb_logger: WandbLogger | None = None,
        write_video: bool = True,
        delete_frames: bool = True,
        heatmap_option: dict[str, Any] | None = None,
    ) -> None:
        self._save_dir = save_dir
        self._name = name
        self._wandb_logger = wandb_logger
        self._write_video = write_video
        self._delete_frames = delete_frames
        self._heatmap_option = heatmap_option or {}
        self._step = 0
        self._frames_dir = f"{self._save_dir}/{self._name}_frames"

        os.makedirs(self._frames_dir, exist_ok=True)

    def __call__(self, data: NDArray[np.float32]) -> None:
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

            if self._wandb_logger is not None:
                self._wandb_logger({f"{self._name}": wandb.Video(name)})

        if self._delete_frames:
            # Delete the frames
            shutil.rmtree(self._frames_dir)

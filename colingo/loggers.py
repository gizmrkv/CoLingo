import os
import shutil
from glob import glob
from typing import Any, Dict, Mapping

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from moviepy.editor import ImageSequenceClip
from numpy.typing import NDArray
from torchtyping import TensorType

import wandb

from .core import RunnerCallback

matplotlib.use("Agg")


class WandbLogger(RunnerCallback):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self.metrics: Dict[str, float] = {}

    def __call__(self, metrics: Mapping[str, Any]) -> None:
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
        wandb.finish()


class HeatmapLogger(RunnerCallback):
    def __init__(
        self,
        save_dir: str,
        name: str,
        wandb_logger: WandbLogger | None = None,
        write_video: bool = True,
        delete_frames: bool = True,
        heatmap_option: Mapping[str, Any] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.name = name
        self.wandb_logger = wandb_logger
        self.write_video = write_video
        self.delete_frames = delete_frames
        self.heatmap_option = heatmap_option or {}
        self.step = 0
        self.frames_dir = f"{self.save_dir}/{self.name}_frames"

        os.makedirs(self.frames_dir, exist_ok=True)

    def __call__(self, data: NDArray[np.float32]) -> None:
        # Save a heatmap frame
        sns.heatmap(data, **self.heatmap_option)
        plt.title(f"{self.name} step: {self.step}")
        plt.savefig(f"{self.frames_dir}/{self.step:0>8}.png")
        plt.clf()

    def on_update(self, step: int) -> None:
        self.step = step

    def on_end(self) -> None:
        if self.write_video:
            # Save a video of the heatmap
            frames = sorted(glob(f"{self.frames_dir}/*.png"))
            name = f"{self.save_dir}/{self.name}.mp4"
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(name)

            if self.wandb_logger is not None:
                self.wandb_logger({f"{self.name}": wandb.Video(name)})

        if self.delete_frames:
            # Delete the frames
            shutil.rmtree(self.frames_dir)


class IntSequenceLanguageLogger:
    def __init__(self, save_dir: str, name: str) -> None:
        self.save_dir = save_dir
        self.name = name
        os.makedirs(f"{self.save_dir}/{self.name}", exist_ok=True)

    def __call__(
        self, step: int, sequence: TensorType[..., int], message: TensorType[..., int]
    ) -> None:
        lines = []
        for seq, msg in zip(sequence, message):
            i = torch.argwhere(msg == 0)
            msg = msg if len(i) == 0 else msg[: i[0, 0]]

            s = str(tuple(seq.tolist()))
            m = str(msg.tolist())
            lines.append(f"{s} -> {m}\n")

        lang = "".join(lines)

        with open(f"{self.save_dir}/{self.name}/{step}.txt", "w") as f:
            f.write(lang)

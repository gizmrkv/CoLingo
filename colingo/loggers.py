import os
import shutil
from glob import glob
from typing import Any, Callable, Dict, Iterable, Mapping

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
        cleanup: bool = False,
        heatmap_option: Mapping[str, Any] | None = None,
        callbacks: Iterable[Callable[[str], None]] | None = None,
    ) -> None:
        self.save_dir = save_dir
        self.cleanup = cleanup
        self.heatmap_option = heatmap_option or {}
        self.callbacks = callbacks or []

        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, step: int, data: NDArray[np.float32]) -> None:
        # Save a heatmap frame
        sns.heatmap(data, **self.heatmap_option)
        plt.title(f"step: {step}")
        plt.savefig(f"{self.save_dir}/{step:0>8}.png")
        plt.clf()

    def on_end(self) -> None:
        # Save a video of the heatmap
        frames = sorted(glob(f"{self.save_dir}/*.png"))
        name = f"{self.save_dir}/video.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(name)

        for callback in self.callbacks:
            callback(name)

        if self.cleanup:
            # Delete the frames
            shutil.rmtree(self.save_dir)


class IntSequenceLanguageLogger:
    def __init__(self, save_dir: str) -> None:
        self.save_dir = save_dir
        os.makedirs(f"{self.save_dir}", exist_ok=True)

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

        with open(f"{self.save_dir}/{step}.txt", "w") as f:
            f.write(lang)

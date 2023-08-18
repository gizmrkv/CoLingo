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
    """
    Callback to log metrics using WandB (Weights and Biases) platform.
    """

    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self.metrics: Dict[str, float] = {}

    def __call__(self, metrics: Mapping[str, Any]) -> None:
        """
        Log metrics to the internal metrics dictionary.

        Args:
            metrics (Mapping[str, Any]): Metrics to log.
        """
        self.metrics.update(metrics)

    def flush(self) -> None:
        """
        Log the metrics to WandB and clear the internal metrics dictionary.
        """
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
    """
    Callback to log heatmap frames and generate a video.
    """

    def __init__(
        self,
        save_dir: str,
        cleanup: bool = False,
        heatmap_option: Mapping[str, Any] | None = None,
        callbacks: Iterable[Callable[[str], None]] | None = None,
    ) -> None:
        """
        Initialize the HeatmapLogger.

        Args:
            save_dir (str): Directory to save heatmap images and video.
            cleanup (bool, optional): Whether to delete heatmap frames and the video after generating the video. Defaults to False.
            heatmap_option (Mapping[str, Any], optional): Options for creating the heatmap using seaborn. Defaults to None.
            callbacks (Iterable[Callable[[str], None]], optional): List of callbacks to be called with the video file name. Defaults to None.
        """

        self.save_dir = save_dir
        self.cleanup = cleanup
        self.heatmap_option = heatmap_option or {}
        self.callbacks = callbacks or []

        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, step: int, data: NDArray[np.float32]) -> None:
        """
        Log a heatmap frame.

        Args:
            step (int): Current step number.
            data (NDArray[np.float32]): Heatmap data.
        """

        sns.heatmap(data, **self.heatmap_option)
        plt.title(f"step: {step}")
        plt.savefig(f"{self.save_dir}/{step:0>8}.png")
        plt.clf()

    def on_end(self) -> None:
        # Generates a video from the saved heatmap frames.
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
    """
    Callback to log integer sequences and corresponding language messages.
    """

    def __init__(self, save_dir: str) -> None:
        """
        Initialize the IntSequenceLanguageLogger.

        Args:
            save_dir (str): Directory to save log files.
        """

        self.save_dir = save_dir
        os.makedirs(f"{self.save_dir}", exist_ok=True)

    def __call__(
        self, step: int, sequence: TensorType[..., int], message: TensorType[..., int]
    ) -> None:
        """
        Log integer sequences and corresponding language messages.

        Args:
            step (int): Current step number.
            sequence (TensorType[..., int]): Integer sequences.
            message (TensorType[..., int]): Corresponding language messages.
        """

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

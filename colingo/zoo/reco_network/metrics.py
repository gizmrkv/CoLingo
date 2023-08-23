import os
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torchtyping import TensorType

from ...analysis import language_similarity, topographic_similarity
from ...game import ReconstructionNetworkSubGameResult
from ...loggers import HeatmapLogger, IntSequenceLanguageLogger
from .agent import MessageAuxiliary
from .loss import Loss


class Metrics:
    def __init__(
        self,
        name: str,
        loss: Loss,
        callbacks: Iterable[Callable[[Dict[str, float]], None]],
    ) -> None:
        self.name = name
        self.loss = loss
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        metrics: Dict[str, float] = {}

        result = next(iter(outputs))

        for name_e, result_e in result.items():
            acc_comps, acc_parts = zip(
                *[
                    acc_comp_part(output_d, result_e.input)
                    for output_d in result_e.outputs.values()
                ]
            )
            acc_comp_mean = fmean(acc_comps)
            acc_part_mean = fmean(acc_parts)
            acc_comp_max = max(acc_comps)
            acc_part_max = max(acc_parts)
            metrics |= {
                f"{self.name}.{name_e}.acc_comp.mean": acc_comp_mean,
                f"{self.name}.{name_e}.acc_part.mean": acc_part_mean,
                f"{self.name}.{name_e}.acc_comp.max": acc_comp_max,
                f"{self.name}.{name_e}.acc_part.max": acc_part_max,
            }

        for callback in self.callbacks:
            callback(metrics)


class TopographicSimilarityMetrics:
    def __init__(
        self, name: str, callbacks: Iterable[Callable[[Dict[str, float]], None]]
    ) -> None:
        self.name = name
        self.callbacks = callbacks

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        metrics: Dict[str, float] = {}
        for name_e, result_e in output.items():
            metrics |= {
                f"{self.name}.{name_e}.topsim": topographic_similarity(
                    result_e.input.cpu().numpy(),
                    result_e.latent.cpu().numpy(),
                    y_processor=drop_padding,  # type: ignore
                )
            }

        metrics[f"{self.name}.topsim.mean"] = fmean(metrics.values())
        for callback in self.callbacks:
            callback(metrics)


class LanguageLogger:
    def __init__(self, save_dir: Path, agent_names: Iterable[str]) -> None:
        self.loggers = {
            name: IntSequenceLanguageLogger(save_dir.joinpath(name))
            for name in agent_names
        }

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        for name_e, result_e in output.items():
            self.loggers[name_e](step, result_e.input, result_e.latent)


class AccuracyHeatmapLogger:
    def __init__(
        self,
        acc_comp_logger: HeatmapLogger,
        acc_part_logger: HeatmapLogger,
    ) -> None:
        self.acc_comp_logger = acc_comp_logger
        self.acc_part_logger = acc_part_logger

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        names = list(output.keys())
        matrix_comp: List[List[float]] = []
        matrix_part: List[List[float]] = []
        for name_e in names:
            comps, parts = [], []
            for name_d in names:
                acc_comp, acc_part = acc_comp_part(
                    output[name_e].outputs[name_d], input
                )
                comps.append(acc_comp)
                parts.append(acc_part)

            matrix_comp.append(comps)
            matrix_part.append(parts)

        for matrix, logger in [
            (matrix_comp, self.acc_comp_logger),
            (matrix_part, self.acc_part_logger),
        ]:
            df = pd.DataFrame(matrix, columns=names, index=names)
            logger(step, df)


class LanguageSimilarityMetrics:
    def __init__(
        self,
        name: str,
        callbacks: Iterable[Callable[[Dict[str, float]], None]],
        heatmap_logger: HeatmapLogger | None = None,
    ) -> None:
        self.name = name
        self.callbacks = callbacks
        self.heatmap_logger = heatmap_logger

    def __call__(
        self,
        step: int,
        input: TensorType[..., int],
        outputs: Iterable[
            Dict[
                str,
                ReconstructionNetworkSubGameResult[
                    TensorType[..., int],
                    TensorType[..., int],
                    MessageAuxiliary,
                    TensorType[..., float],
                ],
            ]
        ],
    ) -> None:
        output = next(iter(outputs))
        names = list(output.keys())

        langs = []
        for name in names:
            langs.append(output[name].latent.cpu().numpy())

        matrix: List[List[float]] = []
        for _ in names:
            matrix.append([0.0] * len(names))

        for i in range(len(names)):
            for j in range(i, len(names)):
                lansim = language_similarity(langs[i], langs[j], processor=drop_padding)
                matrix[i][j] = lansim
                matrix[j][i] = lansim

        if self.heatmap_logger is not None:
            df = pd.DataFrame(matrix, columns=names, index=names)
            self.heatmap_logger(step, df)

        metrics: Dict[str, float] = {}
        for name, lansims in zip(names, matrix):
            metrics[f"{self.name}.{name}.lansim.mean"] = fmean(lansims)

        metrics[f"{self.name}.lansim.mean"] = fmean(metrics.values())

        for callback in self.callbacks:
            callback(metrics)


def acc_comp_part(
    input: TensorType[..., int],
    target: TensorType[..., int],
) -> Tuple[float, float]:
    mark = target == input
    acc_comp = mark.all(dim=-1).float().mean().item()
    acc_part = mark.float().mean(dim=0).mean().item()
    return acc_comp, acc_part


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]

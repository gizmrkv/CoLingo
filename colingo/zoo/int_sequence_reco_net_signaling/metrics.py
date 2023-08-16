import os
from statistics import fmean
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray
from torchtyping import TensorType

from ...analysis import topographic_similarity
from ...game import ReconstructionNetworkSubGameResult
from ...loggers import IntSequenceLanguageLogger
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
                    self.acc(output_d, result_e.input)
                    for output_d in result_e.outputs.values()
                ]
            )
            acc_comp_mean = fmean(acc_comps)
            acc_part_mean = fmean(acc_parts)
            acc_comp_max = max(acc_comps)
            acc_part_max = max(acc_parts)
            metrics |= {
                f"{name_e}.acc_comp.mean": acc_comp_mean,
                f"{name_e}.acc_part.mean": acc_part_mean,
                f"{name_e}.acc_comp.max": acc_comp_max,
                f"{name_e}.acc_part.max": acc_part_max,
            }

        metrics = {f"{self.name}.{k}": v for k, v in metrics.items()}
        for callback in self.callbacks:
            callback(metrics)

    def acc(
        self,
        input: TensorType[..., int],
        target: TensorType[..., int],
    ) -> Tuple[float, float]:
        mark = target == input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc_part = mark.float().mean(dim=0).mean().item()
        return acc_comp, acc_part


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


def drop_padding(x: NDArray[np.int32]) -> NDArray[np.int32]:
    i = np.argwhere(x == 0)
    return x if len(i) == 0 else x[: i[0, 0]]


class LanguageLogger:
    def __init__(self, save_dir: str, agent_names: Iterable[str]) -> None:
        self.loggers = {
            name: IntSequenceLanguageLogger(os.path.join(save_dir, name))
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

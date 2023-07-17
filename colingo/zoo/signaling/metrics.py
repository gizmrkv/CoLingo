from statistics import mean
from typing import Iterable, Sequence

import torch

from ...logger import Logger
from .game import GameResult


class Metrics(Logger):
    def __init__(
        self,
        name: str,
        sender_name: str,
        receiver_names: Sequence[str],
        loggers: Iterable[Logger],
    ) -> None:
        self._name = name
        self._sender_name = sender_name
        self._receiver_names = receiver_names
        self._loggers = loggers

    def log(self, results: list[GameResult | tuple[GameResult, float]]) -> None:
        metrics = []
        for result in results:
            if isinstance(result, tuple):
                metrics.append(self.calc_metrics(*result))
            else:
                metrics.append(self.calc_metrics(result))
        mean_metrics = {k: mean(m[k] for m in metrics) for k in metrics[0]}
        mean_metrics = {
            f"{self._name}.{self._sender_name}->{k}": v for k, v in mean_metrics.items()
        }
        for logger in self._loggers:
            logger.log(mean_metrics)

    def calc_metrics(
        self, result: GameResult, loss: float | None = None
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = loss

        for name_r, output_r in zip(self._receiver_names, result.output_r):
            mark = output_r == result.input
            acc_comp = mark.all(dim=-1).float().mean().item()
            acc = mark.float().mean(dim=0)
            acc_part = acc.mean().item()
            metrics |= {
                f"{name_r}.acc_comp": acc_comp,
                f"{name_r}.acc_part": acc_part,
            }
            metrics |= {f"{name_r}.acc{i}": a.item() for i, a in enumerate(list(acc))}

        return metrics

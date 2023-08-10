from statistics import mean
from typing import Iterable

from ...logger import Logger
from .game import GameResult


class Metrics(Logger):
    def __init__(self, name: str, loggers: Iterable[Logger]):
        self._name = name
        self._loggers = loggers

    def log(self, result: GameResult) -> None:
        metrics: dict[str, float] = {}
        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}
        metrics = {f"{self._name}.{k}": v for k, v in metrics.items()}
        for logger in self._loggers:
            logger.log(metrics)

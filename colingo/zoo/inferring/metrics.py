from statistics import mean
from typing import Iterable

from ...logger import Logger
from .game import GameResult


class Metrics(Logger):
    def __init__(self, name: str, loggers: Iterable[Logger]):
        self._name = name
        self._loggers = loggers

    def log(self, results: list[GameResult | tuple[GameResult, float]]) -> None:
        metrics = []
        for result in results:
            if isinstance(result, tuple):
                metrics.append(self.calc_metrics(*result))
            else:
                metrics.append(self.calc_metrics(result))
        mean_metrics = {k: mean(m[k] for m in metrics) for k in metrics[0]}
        mean_metrics = {f"{self._name}.{k}": v for k, v in mean_metrics.items()}
        for logger in self._loggers:
            logger.log(mean_metrics)

    def calc_metrics(
        self, result: GameResult, loss: float | None = None
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = loss

        mark = result.output == result.input
        acc_comp = mark.all(dim=-1).float().mean().item()
        acc = mark.float().mean(dim=0)
        acc_part = acc.mean().item()
        metrics |= {"acc_comp": acc_comp, "acc_part": acc_part}
        metrics |= {f"acc{i}": a.item() for i, a in enumerate(list(acc))}
        return metrics

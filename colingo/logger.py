from typing import Any, Callable, Iterable, TypeVar

from mypy_extensions import KwArg, VarArg

import wandb

from .core import Callback


class Logger(Callback):
    def log(self, metrics: Any) -> None:
        pass


class WandBLogger(Logger):
    def __init__(self, project: str, name: str | None = None) -> None:
        wandb.init(project=project, name=name)
        self._metrics: dict[str, float] = {}

    def log(self, metrics: dict[str, float]) -> None:
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

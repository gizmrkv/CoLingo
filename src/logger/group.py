from typing import Callable

from .logger import Logger


class Group(Logger):
    def __init__(
        self, *loggers: Logger, metrics: Callable[[list], dict[str, float | int]]
    ):
        self._loggers = loggers
        self._metrics = metrics

        self._logs = []

    def log(self, log):
        self._logs.append(log)

    def on_update(self, step: int):
        metrics = self._metrics(self._logs)
        self._logs.clear()
        for logger in self._loggers:
            logger.log(metrics)

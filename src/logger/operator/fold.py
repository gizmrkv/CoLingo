from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class FoldLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, func: Callable, init=None):
        super().__init__(*loggers)
        self._func = func
        self._prev = init

    def log(self, *args, **kwargs):
        self._prev = self._func(self._prev, *args, **kwargs)
        for logger in self._loggers:
            logger.log(self._prev)


def fold(*loggers: Logger, func: Callable, init=None):
    return FoldLoggerOperator(*loggers, func=func, init=init)


def sum(*loggers: Logger):
    return fold(*loggers, func=lambda x, y: x + y, init=0)


def product(*loggers: Logger):
    return fold(*loggers, func=lambda x, y: x * y, init=1)


def max(*loggers: Logger):
    return fold(*loggers, func=lambda x, y: x if x > y else y, init=-float("inf"))


def min(*loggers: Logger):
    return fold(*loggers, func=lambda x, y: x if x < y else y, init=float("inf"))

from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class FoldFilterLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, pred: Callable, func: Callable, init=None):
        super().__init__(*loggers)
        self._pred = pred
        self._func = func
        self._prev = init

    def log(self, *args, **kwargs):
        self._prev = self._func(self._prev, *args, **kwargs)
        if self._pred(self._prev):
            for logger in self._loggers:
                logger.log(*args, **kwargs)


def fold_filter(*loggers: Logger, pred: Callable, func: Callable, init=None):
    return FoldFilterLoggerOperator(*loggers, pred=pred, init=init, func=func)


def take(*loggers: Logger, n: int):
    return fold_filter(*loggers, pred=lambda x: x <= n, func=lambda x, y: x + 1, init=0)


def skip(*loggers: Logger, n: int):
    return fold_filter(*loggers, pred=lambda x: x > n, func=lambda x, y: x + 1, init=0)


def take_while(*loggers: Logger, pred: Callable):
    return fold_filter(
        *loggers, pred=lambda x: x, func=lambda x, y: x and pred(y), init=True
    )


def skip_while(*loggers: Logger, pred: Callable):
    return fold_filter(
        *loggers, pred=lambda x: not x, func=lambda x, y: x and pred(y), init=True
    )

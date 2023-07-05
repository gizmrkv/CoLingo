from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class OnceLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, pred: Callable):
        super().__init__(*loggers)
        self._pred = pred
        self._once = False

    def log(self, *args, **kwargs):
        if not self._once and self._pred(*args, **kwargs):
            self._once = True
            for logger in self._loggers:
                logger.log(*args, **kwargs)


def once(*loggers: Logger, pred: Callable):
    return OnceLoggerOperator(*loggers, pred=pred)

from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class FilterLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, pred: Callable):
        super().__init__(*loggers)
        self._pred = pred

    def log(self, *args, **kwargs):
        if self._pred(*args, **kwargs):
            for logger in self._loggers:
                logger.log(*args, **kwargs)


def filter(*loggers: Logger, pred: Callable):
    return FilterLoggerOperator(*loggers, pred=pred)

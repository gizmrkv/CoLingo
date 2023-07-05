from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class SelectLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, func: Callable):
        super().__init__(*loggers)
        self._func = func

    def log(self, *args, **kwargs):
        for logger in self._loggers:
            logger.log(self._func(*args, **kwargs))


def select(*loggers: Logger, func: Callable):
    return SelectLoggerOperator(*loggers, func=func)

from typing import Callable

from ..logger import Logger
from .logger_operator import LoggerOperator


class InspectLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger, act: Callable):
        super().__init__(*loggers)
        self._act = act

    def log(self, *args, **kwargs):
        self._act(*args, **kwargs)
        for logger in self._loggers:
            logger.log(*args, **kwargs)


def inspect(*loggers: Logger, act: Callable):
    return InspectLoggerOperator(*loggers, act=act)

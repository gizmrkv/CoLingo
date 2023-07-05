from typing import Iterable

from ..logger import Logger
from .logger_operator import LoggerOperator


class PipeLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger):
        super().__init__(*loggers)

        for i in range(len(loggers) - 1):
            curr, next = loggers[i : i + 2]
            if isinstance(curr, LoggerOperator):
                curr.subscribe(next)

    def log(self, *args, **kwargs):
        self._loggers[0].log(*args, **kwargs)


def pipe(*loggers: Logger):
    return PipeLoggerOperator(*loggers)

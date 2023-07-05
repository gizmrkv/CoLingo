from ..logger import Logger
from .logger_operator import LoggerOperator


class EnumerateLoggerOperator(LoggerOperator):
    def __init__(self, *loggers: Logger):
        super().__init__(*loggers)
        self._count = 0

    def log(self, *args, **kwargs):
        for logger in self._loggers:
            logger.log(self._count, *args, **kwargs)
        self._count += 1


def enumerate(*loggers: Logger):
    return EnumerateLoggerOperator(*loggers)

from typing import Any

from ..logger import Logger


class LoggerOperator(Logger):
    def __init__(self, *loggers: Logger):
        self._loggers = loggers

    def subscribe(self, *loggers: Logger):
        self._loggers = self._loggers + loggers

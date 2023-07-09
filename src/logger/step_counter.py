from ..core import Callback
from .logger import Logger


class StepCounter(Callback):
    def __init__(self, *loggers: Logger, name: str = "steps"):
        self._loggers = loggers
        self._steps = 0
        self._name = name

    def on_update(self, iteration: int):
        self._steps += 1

    def on_end(self):
        for logger in self._loggers:
            logger.log({self._name: self._steps})

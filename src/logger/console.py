from pprint import pprint

from .logger import Logger


class ConsoleLogger(Logger):
    def __init__(self):
        self._logs = {}

    def log(self, log: dict):
        size = len(self._logs)
        self._logs |= log
        assert len(self._logs) == size + len(log), "Duplicate keys found."

    def on_begin(self):
        pprint(self._logs)
        self._logs.clear()

    def on_update(self, step: int):
        pprint(self._logs)
        self._logs.clear()

    def on_end(self):
        pprint(self._logs)
        self._logs.clear()

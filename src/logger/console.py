from pprint import pprint

from .logger import Logger


class ConsoleLogger(Logger):
    def __init__(self):
        super().__init__()
        self._logs = {}

    def log(self, log: dict):
        for key, value in log.items():
            if key in self._logs:
                print(f"Key {key} already exists in logs.")
            self._logs[key] = value

    def on_update(self, iteration: int):
        pprint(self._logs)
        self._logs = {}

    def on_end(self):
        pprint(self._logs)

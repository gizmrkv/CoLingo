import wandb

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, *args, **kwargs):
        self._logs = {}

        wandb.init(*args, **kwargs)

    def log(self, log: dict):
        size = len(self._logs)
        self._logs |= log
        assert size + len(log) == len(self._logs), "Duplicate keys in log"

    def on_begin(self):
        wandb.log(self._logs)
        self._logs.clear()

    def on_update(self, iteration: int):
        wandb.log(self._logs)
        self._logs.clear()

    def on_end(self):
        wandb.log(self._logs)
        self._logs.clear()

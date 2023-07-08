import wandb

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, *args, **kwargs):
        self._logs = {}

        wandb.init(*args, **kwargs)

    def log(self, log: dict):
        self._logs |= log

    def on_update(self, iteration: int):
        wandb.log(self._logs)
        self.logs = {}

    def on_end(self):
        wandb.log(self._logs)
        self.logs = {}

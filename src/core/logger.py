from abc import ABC, abstractmethod
from pprint import pprint

import wandb

from ..core.callback import Callback


class Logger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = {}

    def log(self, logs: dict, flush: bool = False):
        self.logs |= logs
        if flush:
            self.flush()

    def flush(self):
        pass

    def on_post_update(self):
        self.flush()


class ConsoleLogger(Logger):
    def flush(self):
        pprint(self.logs)
        self.logs = {}


class WandBLogger(Logger):
    def __init__(self, **kwargs):
        super().__init__()
        wandb.init(**kwargs)

    def flush(self):
        wandb.log(self.logs)
        self.logs = {}

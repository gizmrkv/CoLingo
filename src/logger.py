from pprint import pprint

import wandb


class Logger:
    def log(self, logs: dict):
        pass


class ConsoleLogger(Logger):
    def log(self, logs: dict):
        pprint(logs)


class WandBLogger(Logger):
    def __init__(self, **kwargs):
        super().__init__()
        wandb.init(**kwargs)

    def log(self, logs: dict):
        wandb.log(logs)

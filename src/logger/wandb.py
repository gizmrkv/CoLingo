import wandb

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__()
        wandb.init(*args, **kwargs)

    def flush(self):
        wandb.log(self.logs)
        self.logs = {}

import wandb

from .logger import Logger


class WandBLogger(Logger):
    def __init__(self, **kwargs):
        super().__init__()
        wandb.init(**kwargs)

    def flush(self):
        wandb.log(self.logs)
        self.logs = {}

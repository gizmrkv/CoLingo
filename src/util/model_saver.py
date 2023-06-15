import os

import torch as th

from ..core import Callback


class ModelSaver(Callback):
    def __init__(
        self,
        models: dict[str, th.nn.Module],
        path: str,
    ):
        super().__init__()
        self.models = models
        self.path = path

    def on_update(self, iteration: int):
        for model_name, model in self.models.items():
            save_dir = f"{self.path}/{model_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            th.save(
                model,
                f"{save_dir}/{iteration}.pth",
            )

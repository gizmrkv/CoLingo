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

        for model_name in self.models:
            save_dir = f"{self.path}/{model_name}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def on_begin(self):
        for model_name, model in self.models.items():
            th.save(model, f"{self.path}/{model_name}/begin.pth")

    def on_update(self, step: int):
        for model_name, model in self.models.items():
            th.save(model, f"{self.path}/{model_name}/{step}.pth")

    def on_end(self):
        for model_name, model in self.models.items():
            th.save(model, f"{self.path}/{model_name}/end.pth")

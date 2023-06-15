from typing import Iterable

import torch as th

from ..core import Callback


class ModelInitializer(Callback):
    def __init__(
        self,
        model: th.nn.Module | Iterable[th.nn.Module],
    ):
        super().__init__()
        self.models = [model] if isinstance(model, th.nn.Module) else model

    def on_begin(self):
        for model in self.models:
            model.apply(init_weights)

    def on_update(self, iteration: int):
        for model in self.models:
            model.apply(init_weights)


def init_weights(m):
    if isinstance(m, (th.nn.Linear, th.nn.Conv2d)):
        th.nn.init.kaiming_uniform_(m.weight)
        th.nn.init.zeros_(m.bias)
    elif isinstance(m, (th.nn.RNN, th.nn.LSTM, th.nn.GRU)):
        th.nn.init.kaiming_uniform_(m.weight_ih_l0)
        th.nn.init.kaiming_uniform_(m.weight_hh_l0)
        th.nn.init.zeros_(m.bias_ih_l0)
        th.nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, th.nn.Embedding):
        th.nn.init.kaiming_uniform_(m.weight)
    elif isinstance(
        m, (th.nn.LayerNorm, th.nn.BatchNorm1d, th.nn.BatchNorm2d, th.nn.BatchNorm3d)
    ):
        th.nn.init.constant_(m.weight, 1)
        th.nn.init.constant_(m.bias, 0)

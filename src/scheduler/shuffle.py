import random

from ..core import Callback
from .scheduler import Scheduler


class ShuffleScheduler(Scheduler):
    def __init__(
        self,
        callbacks: list[Callback],
    ):
        super().__init__()
        self.callbacks = callbacks
        self.order = list(range(len(self.callbacks)))

    def on_pre_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_pre_update(iteration)

        random.shuffle(self.order)

    def on_update(self, iteration: int):
        for i in self.order:
            self.callbacks[i].on_update(iteration)

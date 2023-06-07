import random

from ..core import Callback


class ShuffleScheduler(Callback):
    def __init__(
        self,
        callbacks: list[Callback],
    ):
        super().__init__()
        self.callbacks = callbacks
        self.order = list(range(len(self.callbacks)))

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_pre_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_pre_update()

        random.shuffle(self.order)

    def on_update(self, iteration: int):
        for i in self.order:
            self.callbacks[i].on_update()

    def on_post_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_post_update()

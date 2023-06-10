from typing import Iterable

from ..core import Callback


class Scheduler(Callback):
    def __init__(self, callback: Callback | Iterable[Callback]):
        super().__init__()
        self.callbacks = [callback] if isinstance(callback, Callback) else callback

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_pre_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_update(iteration)

    def on_post_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_post_update(iteration)

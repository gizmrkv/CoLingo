import math
import random
from typing import Callable

from ..core import Callback


class CustomScheduler(Callback):
    def __init__(
        self,
        callback: Callback | list[Callback],
        frequency: Callable[[int], float],
        randomly: bool = False,
    ):
        super().__init__()
        self.callbacks = [callback] if isinstance(callback, Callback) else callback
        self.freq_lambda = frequency
        self.randomly = randomly

        self.call_count = 0
        if not randomly:
            self.freq_pool = 0

        self.run_count = 0

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_pre_update(self, iteration: int):
        freq = self.freq_lambda(self.call_count)
        self.call_count += 1
        if self.randomly:
            self.run_count = int(random.random() < freq)
        else:
            self.freq_pool, self.run_count = math.modf(self.freq_pool + freq)
            self.run_count = int(self.run_count)

        if self.run_count > 0:
            for callback in self.callbacks:
                callback.on_pre_update()

    def on_update(self, iteration: int):
        for _ in range(self.run_count):
            for callback in self.callbacks:
                callback.on_update()

    def on_post_update(self, iteration: int):
        if self.run_count > 0:
            for callback in self.callbacks:
                callback.on_post_update()

        self.run_count = 0

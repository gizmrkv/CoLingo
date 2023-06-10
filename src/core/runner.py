from typing import Iterable

import tqdm

from .callback import Callback


class Runner:
    def __init__(self, callback: Callback | Iterable[Callback], use_tqdm: bool = True):
        self.callbacks = [callback] if isinstance(callback, Callback) else callback
        self.use_tqdm = use_tqdm

    def run(self, n_iterations: int):
        for callback in self.callbacks:
            callback.on_begin()

        rg = range(n_iterations)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for iter in rg:
            for callback in self.callbacks:
                callback.on_pre_update(iter)

            for callback in self.callbacks:
                callback.on_update(iter)

            for callback in self.callbacks:
                callback.on_post_update(iter)

        for callback in self.callbacks:
            callback.on_end()

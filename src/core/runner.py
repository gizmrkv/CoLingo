from typing import Iterable

import tqdm

from .callback import Callback


class Runner:
    def __init__(self, callback: Callback | Iterable[Callback]):
        self.callbacks = [callback] if isinstance(callback, Callback) else callback

    def run(self, n_iterations: int):
        for callback in self.callbacks:
            callback.on_begin()

        for iter in tqdm.tqdm(range(n_iterations)):
            for callback in self.callbacks:
                callback.on_pre_update(iter)

            for callback in self.callbacks:
                callback.on_update(iter)

            for callback in self.callbacks:
                callback.on_post_update(iter)

        for callback in self.callbacks:
            callback.on_end()

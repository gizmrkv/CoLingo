from typing import Callable

import tqdm

from .callback import Callback


class Runner:
    def __init__(
        self,
        *callbacks: Callback,
        early_stop: Callable[[int], bool] | None = None,
        use_tqdm: bool = True,
    ):
        self.callbacks = callbacks
        self.early_stop = (lambda _: False) if early_stop is None else early_stop
        self.use_tqdm = use_tqdm

    def run(self, n_steps: int):
        for callback in self.callbacks:
            callback.on_begin()

        rg = range(n_steps)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for step in rg:
            if self.early_stop():
                for callback in self.callbacks:
                    callback.on_early_stop(step)
                break
            else:
                for callback in self.callbacks:
                    callback.on_update(step)

        for callback in self.callbacks:
            callback.on_end()

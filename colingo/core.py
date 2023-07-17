from abc import ABC
from typing import Callable, Iterable

import tqdm


class Callback(ABC):
    def on_begin(self) -> None:
        pass

    def on_update(self, step: int) -> None:
        pass

    def on_end(self) -> None:
        pass


class Runner:
    def __init__(
        self,
        callbacks: Iterable[Callback],
        early_stop: Callable[[int], bool] | None = None,
        use_tqdm: bool = True,
    ):
        self._callbacks = callbacks
        self._early_stop = (lambda _: False) if early_stop is None else early_stop
        self._use_tqdm = use_tqdm

    def run(self, n_steps: int) -> None:
        for callback in self._callbacks:
            callback.on_begin()

        rg: Iterable[int] = range(n_steps)
        if self._use_tqdm:
            rg = tqdm.tqdm(rg)

        for step in rg:
            if self._early_stop(step):
                break

            for callback in self._callbacks:
                callback.on_update(step)

        for callback in self._callbacks:
            callback.on_end()

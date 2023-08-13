from typing import Iterable

import tqdm


class RunnerCallback:
    def on_begin(self) -> None:
        pass

    def on_update(self, step: int) -> None:
        pass

    def on_end(self) -> None:
        pass


class EarlyStopper:
    def stop(self, step: int) -> bool:
        return False


class Runner:
    def __init__(
        self,
        callbacks: Iterable[RunnerCallback],
        stopper: EarlyStopper | None = None,
        use_tqdm: bool = True,
    ):
        self.callbacks = callbacks
        self.use_tqdm = use_tqdm
        self.stopper = stopper or EarlyStopper()

    def run(self, n_steps: int) -> None:
        for callback in self.callbacks:
            callback.on_begin()

        rg: Iterable[int] = range(n_steps)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for step in rg:
            if self.stopper.stop(step):
                break

            for callback in self.callbacks:
                callback.on_update(step)

        for callback in self.callbacks:
            callback.on_end()

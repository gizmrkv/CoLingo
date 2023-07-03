from typing import Callable, Iterable

import tqdm

from .callback import Callback


class Runner:
    def __init__(
        self,
        task: Callback | Iterable[Callback],
        early_stop: Callable[[int], bool] | None = None,
        use_tqdm: bool = True,
    ):
        self.tasks = [task] if isinstance(task, Callback) else task
        self.early_stop = lambda _: False if early_stop is None else early_stop
        self.use_tqdm = use_tqdm

    def run(self, n_iterations: int):
        for task in self.tasks:
            task.on_begin()

        rg = range(n_iterations)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for iter in rg:
            if self.early_stop(iter):
                for task in self.tasks:
                    task.on_early_stop(iter)
                break
            else:
                for task in self.tasks:
                    task.on_pre_update(iter)

                for task in self.tasks:
                    task.on_update(iter)

                for task in self.tasks:
                    task.on_post_update(iter)

        for task in self.tasks:
            task.on_end()

from typing import Iterable, List

import tqdm

from .abstract import Stoppable, Task


class TaskRunner:
    def __init__(
        self,
        tasks: List[Task],
        stopper: Stoppable | None = None,
        use_tqdm: bool = True,
    ):
        self.tasks = tasks
        self.tasks.sort(key=lambda task: task.priority(), reverse=True)
        self.use_tqdm = use_tqdm

        class DefaultStopper(Stoppable):
            def stop(self, step: int) -> bool:
                return False

        self.stopper = stopper or DefaultStopper()

    def run(self, n_steps: int) -> None:
        for task in self.tasks:
            task.on_begin()

        rg: Iterable[int] = range(n_steps)
        if self.use_tqdm:
            rg = tqdm.tqdm(rg)

        for step in rg:
            if self.stopper.stop(step):
                break

            for task in self.tasks:
                task.on_update(step)

        for task in self.tasks:
            task.on_end()

from typing import Iterable

from .callback import Callback


class TaskRunner:
    def __init__(self, tasks: Iterable[Callback]):
        self.tasks = tasks

    def run(self, n_iterations: int):

        for task in self.tasks:
            task.on_begin()

        for _ in range(n_iterations):
            for task in self.tasks:
                task.on_update()

        for task in self.tasks:
            task.on_end()

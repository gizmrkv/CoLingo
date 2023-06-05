from typing import Iterable

import tqdm

from .callback import Callback


class TaskRunner:
    """
    This class manages and executes a series of tasks represented as callbacks.
    It provides an interface to run a certain number of iterations over these tasks.

    Args:
        tasks (Iterable[Callback]): The tasks to be run. Each task is represented as a callback.

    Methods:
        run(n_iterations: int): Runs the specified number of iterations over the tasks.
    """

    def __init__(self, task: Callback | Iterable[Callback]):
        self.tasks = [task] if isinstance(task, Callback) else task

    def run(self, n_iterations: int):
        """
        Runs the specified number of iterations over the tasks. During each iteration,
        each task is updated. At the beginning and end of the run, the `on_begin` and
        `on_end` methods of each task are called, respectively.

        Args:
            n_iterations (int): The number of iterations to run.
        """
        for task in self.tasks:
            task.on_begin()

        for iter in tqdm.tqdm(range(n_iterations)):
            for task in self.tasks:
                task.on_pre_update(iter)

            for task in self.tasks:
                task.on_update(iter)

            for task in self.tasks:
                task.on_post_update(iter)

        for task in self.tasks:
            task.on_end()

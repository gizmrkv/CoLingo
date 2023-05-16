from typing import Iterable

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
    def __init__(self, tasks: Iterable[Callback]):
        self.tasks = tasks

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

        for _ in range(n_iterations):
            for task in self.tasks:
                task.on_update()

        for task in self.tasks:
            task.on_end()

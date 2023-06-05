import math
import random
from typing import Callable

from ..core import Callback


class CustomTaskScheduler(Callback):
    def __init__(
        self,
        task: Callback | list[Callback],
        frequency: Callable[[int], float],
        randomly: bool = False,
    ):
        super().__init__()
        self.tasks = [task] if isinstance(task, Callback) else task
        self.freq_lambda = frequency
        self.randomly = randomly

        self.call_count = 0
        if not randomly:
            self.freq_pool = 0

        self.run_count = 0

    def on_begin(self):
        for task in self.tasks:
            task.on_begin()

    def on_end(self):
        for task in self.tasks:
            task.on_end()

    def on_pre_update(self):
        freq = self.freq_lambda(self.call_count)
        self.call_count += 1
        if self.randomly:
            self.run_count = int(random.random() < freq)
        else:
            self.freq_pool, self.run_count = math.modf(self.freq_pool + freq)
            self.run_count = int(self.run_count)

        if self.run_count > 0:
            for task in self.tasks:
                task.on_pre_update()

    def on_update(self):
        for _ in range(self.run_count):
            for task in self.tasks:
                task.on_update()

    def on_post_update(self):
        if self.run_count > 0:
            for task in self.tasks:
                task.on_post_update()

        self.run_count = 0

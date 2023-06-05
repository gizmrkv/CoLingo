import random

from ..core import Callback


class ShuffleTaskScheduler(Callback):
    def __init__(
        self,
        tasks: list[Callback],
    ):
        super().__init__()
        self.tasks = tasks
        self.order = list(range(len(self.tasks)))

    def on_begin(self):
        for task in self.tasks:
            task.on_begin()

    def on_end(self):
        for task in self.tasks:
            task.on_end()

    def on_pre_update(self, iteration: int):
        for task in self.tasks:
            task.on_pre_update()

        random.shuffle(self.order)

    def on_update(self, iteration: int):
        for i in self.order:
            self.tasks[i].on_update()

    def on_post_update(self, iteration: int):
        for task in self.tasks:
            task.on_post_update()

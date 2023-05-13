import math
import random

from .callback import Callback


class LinearTaskScheduler(Callback):
    def __init__(
        self,
        task: Callback,
        trans_begin: int,
        trans_end: int,
        begin_rate: float = 1.0,
        end_rate: float = 0.0,
        run_on_begin: bool = True,
        run_on_end: bool = True,
    ):
        assert 0 <= trans_begin < trans_end

        super().__init__()
        self.task = task
        self.trans_begin = trans_begin
        self.trans_end = trans_end
        self.begin_rate = begin_rate
        self.end_rate = end_rate
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end

        self._count = 0
        self._pool = 0

    def on_begin(self):
        if self.run_on_begin:
            return self.task.on_begin()

    def on_update(self):
        if self._count < self.trans_begin:
            rate = self.begin_rate
        elif self._count > self.trans_end:
            rate = self.end_rate
        else:
            rate = (self.end_rate - self.begin_rate) / (
                self.trans_end - self.trans_begin
            ) * (self._count - self.trans_begin) + self.begin_rate

        self._count += 1
        self._pool, overflows = math.modf(self._pool + rate)
        for _ in range(int(overflows)):
            self.task.on_update()

    def on_end(self):
        if self.run_on_end:
            return self.task.on_end()

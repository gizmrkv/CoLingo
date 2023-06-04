import math
import random

from ..core import Callback


class LinearTaskScheduler(Callback):
    """
    This class is a task scheduler that linearly changes the task execution frequency over a specified period.

    Args:
        task (Callback): The task to be scheduled.
        trans_begin (int): The period at which the change begins.
        trans_duration (int): The duration of the change. Defaults to 1.
        begin_rate (float, optional): The execution frequency at the start of the change period. Defaults to 1.0.
        end_rate (float, optional): The execution frequency at the end of the change period. Defaults to 0.0.
        run_on_begin (bool, optional): A flag that determines if the task should be run at the beginning of the change period. Defaults to True.
        run_on_end (bool, optional): A flag that determines if the task should be run at the end of the change period. Defaults to True.
        randomly (bool, optional): A flag that determines if the task should be run randomly within the change period. Defaults to False.
    """

    def __init__(
        self,
        task: Callback,
        begin_rate: float,
        end_rate: float,
        trans_begin: int,
        trans_duration: int = 1,
        run_on_begin: bool = True,
        run_on_end: bool = True,
        randomly: bool = False,
    ):
        assert 0 <= trans_begin and 1 <= trans_duration

        super().__init__()
        self.task = task
        self.trans_begin = trans_begin
        self.trans_end = trans_begin + trans_duration
        self.begin_rate = begin_rate
        self.end_rate = end_rate
        self.run_on_begin = run_on_begin
        self.run_on_end = run_on_end
        self.randomly = randomly

        self._count = 0
        self._pool = 0

    def on_begin(self):
        """
        This method is called at the start of the scheduling. If `run_on_begin` is True, this method calls the `on_begin` method of the task.
        """
        if self.run_on_begin:
            self.task.on_begin()

    def on_update(self):
        """
        This method updates the execution rate according to the linear schedule.
        If `randomly` is True, it executes the task with a probability equal to the current rate.
        Otherwise, it executes the task as many times as the rate rounded down to the nearest integer.
        """
        if self._count < self.trans_begin:
            rate = self.begin_rate
        elif self._count > self.trans_end:
            rate = self.end_rate
        else:
            rate = (self.end_rate - self.begin_rate) / (
                self.trans_end - self.trans_begin
            ) * (self._count - self.trans_begin) + self.begin_rate

        self._count += 1
        if self.randomly:
            if random.random() < rate:
                self.task.on_update()
        else:
            self._pool, overflows = math.modf(self._pool + rate)
            for _ in range(int(overflows)):
                self.task.on_update()

    def on_end(self):
        """
        This method is called at the end of the scheduling. If `run_on_end` is True, this method calls the `on_end` method of the task.
        """
        if self.run_on_end:
            self.task.on_end()

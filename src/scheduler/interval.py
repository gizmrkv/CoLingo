from ..core import Callback
from .scheduler import Scheduler


class IntervalScheduler(Scheduler):
    def __init__(
        self,
        callback: Callback | list[Callback],
        interval: int,
        offset: int = 0,
    ):
        assert interval > 0 and offset >= 0

        super().__init__(callback)
        self.interval = interval
        self.offset = offset

    def on_update(self, iteration: int):
        diff = iteration - self.offset
        if diff >= 0 and diff % self.interval == 0:
            for callback in self.callbacks:
                callback.on_update(iteration)

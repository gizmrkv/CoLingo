from ..core import Callback
from .scheduler import Scheduler


class IntervalScheduler(Scheduler):
    def __init__(
        self,
        callback: Callback | list[Callback],
        interval: int,
        offset: int = 0,
        in_order: bool = False,
    ):
        assert interval > 0 and offset >= 0, "interval and offset must be positive"

        super().__init__(callback)
        self.interval = interval
        self.offset = offset
        self.in_order = in_order

        if in_order:
            self._next = 0

    def on_update(self, iteration: int):
        diff = iteration - self.offset
        if diff >= 0 and diff % self.interval == 0:
            if self.in_order:
                self.callbacks[self._next].on_update(iteration)
                self._next = (self._next + 1) % len(self.callbacks)
            else:
                for callback in self.callbacks:
                    callback.on_update(iteration)

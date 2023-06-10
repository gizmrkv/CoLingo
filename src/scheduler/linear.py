from ..core import Callback
from .scheduler import Scheduler
from .util import lambda_scheduler


class LinearScheduler(Scheduler):
    def __init__(
        self,
        callback: Callback | list[Callback],
        total_step: int,
        first_freq: float = 1.0,
        final_freq: float = 0.0,
        randomly: bool = False,
    ):
        super().__init__(callback)
        self.total_step = total_step
        self.first_freq = first_freq
        self.final_freq = final_freq
        self.randomly = randomly
        self.scheduler = lambda_scheduler(self.f, randomly)

    def on_update(self, iteration: int):
        for _ in range(next(self.scheduler)):
            for callback in self.callbacks:
                callback.on_update(iteration)

    def f(self, i: int) -> float:
        return (
            self.final_freq
            if i >= self.total_step
            else self.first_freq
            + i / self.total_step * (self.final_freq - self.first_freq)
        )

from ..core import Callback
from .custom import CustomScheduler


class LinearScheduler(CustomScheduler):
    def __init__(
        self,
        callback: Callback | list[Callback],
        total_step: int,
        first_freq: float = 1.0,
        final_freq: float = 0.0,
        randomly: bool = False,
    ):
        self.total_step = total_step
        self.first_freq = first_freq
        self.final_freq = final_freq
        self.frequency = (
            lambda count: self.final_freq
            if count >= self.total_step
            else self.first_freq
            + count / self.total_step * (self.final_freq - self.first_freq)
        )
        super().__init__(callback, self.frequency, randomly)

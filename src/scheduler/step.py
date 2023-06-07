from ..core import Callback
from .custom import CustomScheduler


class StepScheduler(CustomScheduler):
    def __init__(
        self,
        callback: Callback | list[Callback],
        step: int,
        first_freq: float = 1.0,
        final_freq: float = 0.0,
        randomly: bool = False,
    ):
        self.step = step
        self.first_freq = first_freq
        self.final_freq = final_freq
        self.frequency = (
            lambda count: self.first_freq if count < self.step else self.final_freq
        )
        super().__init__(callback, self.frequency, randomly)

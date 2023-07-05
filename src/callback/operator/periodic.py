from ..callback import Callback
from .callback_operator import CallbackOperator


class PeriodicCallbackOperator(CallbackOperator):
    def __init__(self, *callbacks: Callback, period: int, offset: int):
        super().__init__(*callbacks)
        self._period = period
        self._offset = offset

        self._cnt = 0

    def on_update(self, iteration: int):
        diff = self._cnt - self._offset
        if diff >= 0 and diff % self._period == 0:
            for callback in self._callbacks:
                callback.on_update(iteration)
        self._cnt += 1


def periodic(*callbacks: Callback, period: int, offset: int = 0):
    return PeriodicCallbackOperator(*callbacks, offset=offset, period=period)

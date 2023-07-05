from ..callback import Callback
from .callback_operator import CallbackOperator


class RangeCallbackOperator(CallbackOperator):
    def __init__(self, *callbacks: Callback, start: int, count: int):
        super().__init__(*callbacks)
        self._start = start
        self._count = count

        self._cnt = 0

    def on_update(self, iteration: int):
        if self._cnt >= self._start and self._cnt < self._start + self._count:
            for callback in self._callbacks:
                callback.on_update(iteration)
        self._cnt += 1


def range(*callbacks: Callback, start: int, count: int):
    return RangeCallbackOperator(*callbacks, start=start, count=count)

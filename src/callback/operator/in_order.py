from ..callback import Callback
from .callback_operator import CallbackOperator


class InOrderCallbackOperator(CallbackOperator):
    def __init__(self, *callbacks: Callback, intervals: list[int], loop: bool):
        assert len(callbacks) == len(
            intervals
        ), "The length of callbacks and intervals must be the same."
        super().__init__(*callbacks)
        self._intervals = intervals
        self._loop = loop

        self._len = len(callbacks)
        self._cnt = 0
        self._idx = 0

    def on_update(self, iteration: int):
        if self._cnt < self._intervals[self._idx]:
            self._cnt += 1
        else:
            self._idx += 1
            self._cnt = 1

        if self._loop and self._idx >= self._len:
            self._idx = 0

        if self._idx < self._len:
            self._callbacks[self._idx].on_update(iteration)


def in_order(
    *callbacks: Callback, intervals: list[int] | None = None, loop: bool = True
):
    if intervals is None:
        intervals = [1] * len(callbacks)
    return InOrderCallbackOperator(*callbacks, intervals=intervals, loop=loop)

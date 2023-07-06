from ..subject import Subject
from ..subscriber import Subscriber


class InOrderOperator(Subject):
    def __init__(self, *subscribers: Subscriber, intervals: list[int], loop: bool):
        super().__init__(*subscribers)
        self._intervals = intervals
        self._loop = loop

        self._len = len(subscribers)
        self._cnt = 0
        self._idx = 0

    def on_next(self, *args, **kwargs):
        if self._cnt < self._intervals[self._idx]:
            self._cnt += 1
        else:
            self._idx += 1
            self._cnt = 1

        if self._loop and self._idx >= self._len:
            self._idx = 0

        if self._idx < self._len:
            self._subscribers[self._idx].on_next(*args, **kwargs)


def in_order(
    *subscribers: Subscriber, intervals: list[int] | None = None, loop: bool = True
):
    if intervals is None:
        intervals = [1] * len(subscribers)
    return InOrderOperator(*subscribers, intervals=intervals, loop=loop)

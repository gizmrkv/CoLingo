from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class TakeWhileOperator(Subject):
    def __init__(self, *subscribers: Subscriber, pred: Callable):
        super().__init__(*subscribers)
        self._pred = pred
        self._prev = True

    def on_next(self, *args, **kwargs):
        next = self._pred(*args, **kwargs)
        if self._prev and next:
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)

        if self._prev and not next:
            for subscriber in self._subscribers:
                subscriber.on_completed()

        self._prev &= next


def take_while(*subscribers: Subscriber, pred: Callable):
    return TakeWhileOperator(*subscribers, pred=pred)


def take(*subscribers: Subscriber, n: int):
    count = 0

    def localpred(*args, **kwargs):
        nonlocal count
        count += 1
        return count <= n

    return take_while(*subscribers, pred=localpred)

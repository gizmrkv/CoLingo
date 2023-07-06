from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class SkipWhileOperator(Subject):
    def __init__(self, *subscribers: Subscriber, pred: Callable):
        super().__init__(*subscribers)
        self._pred = pred
        self._prev = True

    def on_next(self, *args, **kwargs):
        next = self._pred(*args, **kwargs)
        if not (self._prev and next):
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)

        self._prev &= next


def skip_while(*subscribers: Subscriber, pred: Callable):
    return SkipWhileOperator(*subscribers, pred=pred)


def skip(*subscribers: Subscriber, n: int):
    count = 0

    def pred(*args, **kwargs):
        nonlocal count
        count += 1
        return count <= n

    return skip_while(*subscribers, pred=pred)

from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class FirstOperator(Subject):
    def __init__(self, *subscribers: Subscriber, pred: Callable):
        super().__init__(*subscribers)
        self._pred = pred
        self._first = True

    def on_next(self, *args, **kwargs):
        if self._first and self._pred(*args, **kwargs):
            self._first = False
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)
                subscriber.on_completed()


def first(*subscribers: Subscriber, pred: Callable):
    return FirstOperator(*subscribers, pred=pred)

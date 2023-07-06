from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class LastOperator(Subject):
    def __init__(self, *subscribers: Subscriber, pred: Callable):
        super().__init__(*subscribers)
        self._pred = pred
        self._last = True

    def on_next(self, *args, **kwargs):
        if self._last and not self._pred(*args, **kwargs):
            self._last = False
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)
                subscriber.on_completed()


def last(*subscribers: Subscriber, pred: Callable):
    return LastOperator(*subscribers, pred=pred)

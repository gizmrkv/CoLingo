from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class WhereOperator(Subject):
    def __init__(self, *subscribers: Subscriber, pred: Callable):
        super().__init__(*subscribers)
        self._pred = pred

    def on_next(self, *args, **kwargs):
        if self._pred(*args, **kwargs):
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)


def where(*subscribers: Subscriber, pred: Callable):
    return WhereOperator(*subscribers, pred=pred)

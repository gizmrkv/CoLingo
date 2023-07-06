from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class SelectOperator(Subject):
    def __init__(self, *subscribers: Subscriber, func: Callable):
        super().__init__(*subscribers)
        self._func = func

    def on_next(self, *args, **kwargs):
        for subscriber in self._subscribers:
            subscriber.on_next(self._func(*args, **kwargs))


def select(*subscribers: Subscriber, func: Callable):
    return SelectOperator(*subscribers, func=func)

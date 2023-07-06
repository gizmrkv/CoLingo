from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class FoldOperator(Subject):
    def __init__(self, *subscribers: Subscriber, init, func: Callable):
        super().__init__(*subscribers)
        self._accum = init
        self._func = func

    def on_next(self, *args, **kwargs):
        self._accum = self._func(self._accum, *args, **kwargs)
        for subscriber in self._subscribers:
            subscriber.on_next(self._accum, *args, **kwargs)


def fold(*subscribers: Subscriber, init, func: Callable):
    return FoldOperator(*subscribers, init=init, func=func)


def sum(*subscribers: Subscriber):
    return fold(*subscribers, init=0, func=lambda x, y: x + y)


def product(*subscribers: Subscriber):
    return fold(*subscribers, init=1, func=lambda x, y: x * y)

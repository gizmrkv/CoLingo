import random
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


def range(*subscribers: Subscriber, start: int, stop: int):
    count = start

    def localpred(*args, **kwargs):
        nonlocal count
        count += 1
        return start <= count < stop

    return where(*subscribers, pred=localpred)


def periodic(*subscribers: Subscriber, period: int, offset: int = 0):
    count = 0

    def localpred(*args, **kwargs):
        nonlocal count
        diff = count - offset
        count += 1
        return diff >= 0 and diff % period == 0

    return where(*subscribers, pred=localpred)


def randomly(*subscribers: Subscriber, p: float):
    def localpred(*args, **kwargs):
        return random.random() < p

    return where(*subscribers, pred=localpred)

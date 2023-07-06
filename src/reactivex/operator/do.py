from typing import Callable

from ..subject import Subject
from ..subscriber import Subscriber


class DoOperator(Subject):
    def __init__(self, *subscribers: Subscriber, act: Callable):
        super().__init__(*subscribers)
        self._act = act

    def on_next(self, *args, **kwargs):
        self._act(*args, **kwargs)
        for subscriber in self._subscribers:
            subscriber.on_next(*args, **kwargs)


def do(*subscribers: Subscriber, act: Callable):
    return DoOperator(*subscribers, act=act)

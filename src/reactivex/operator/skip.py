from ..subject import Subject
from ..subscriber import Subscriber


class SkipOperator(Subject):
    def __init__(self, *subscribers: Subscriber, count: int):
        super().__init__(*subscribers)
        self._count = count
        self._skipped = 0

    def on_next(self, *args, **kwargs):
        if self._skipped >= self._count:
            for subscriber in self._subscribers:
                subscriber.on_next(*args, **kwargs)
        else:
            self._skipped += 1


def skip(*subscribers: Subscriber, count: int):
    return SkipOperator(*subscribers, count=count)

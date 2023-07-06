from ..subject import Subject
from ..subscriber import Subscriber


class PipeOperator(Subject):
    def __init__(self, *subjects: Subject):
        super().__init__(*subjects)
        for i in range(len(subjects) - 1):
            curr, next = subjects[i : i + 2]
            if isinstance(curr, Subject):
                curr.subscribe(next)

    def on_next(self, *args, **kwargs):
        self._subscribers[0].on_next(*args, **kwargs)

    def on_completed(self):
        self._subscribers[0].on_completed()

    def on_error(self, error):
        self._subscribers[0].on_error(error)

    def subscribe(self, *subscribers: Subscriber):
        self._subscribers[-1].subscribe(*subscribers)


def pipe(*subjects: Subject):
    return PipeOperator(*subjects)

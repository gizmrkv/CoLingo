from ..subject import Subject
from ..subscriber import Subscriber


class MergeOperator(Subject):
    def __init__(self, *subjects: Subject, target: Subscriber):
        super().__init__(*subjects)
        self._target = target
        for subject in subjects:
            subject.subscribe(target)

    def subscribe(self, *subscribers: Subscriber):
        self._target.subscribe(*subscribers)


def merge(*subjects: Subject, target: Subscriber):
    return MergeOperator(*subjects, target=target)

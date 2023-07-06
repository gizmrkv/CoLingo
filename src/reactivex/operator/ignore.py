from ..subject import Subject
from ..subscriber import Subscriber


class IgnoreOperator(Subject):
    def __init__(self, *subscribers: Subscriber):
        super().__init__(*subscribers)

    def on_next(self, *args, **kwargs):
        pass


def ignore(*subscribers: Subscriber):
    return IgnoreOperator(*subscribers)

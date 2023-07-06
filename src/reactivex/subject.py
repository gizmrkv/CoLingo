from .publisher import Publisher
from .subscriber import Subscriber


class Subject(Subscriber, Publisher):
    def __init__(self, *subscribers: Subscriber):
        self._subscribers = subscribers

    def on_next(self, *args, **kwargs):
        for subscriber in self._subscribers:
            subscriber.on_next(*args, **kwargs)

    def on_completed(self):
        for subscriber in self._subscribers:
            subscriber.on_completed()

    def on_error(self, error):
        for subscriber in self._subscribers:
            subscriber.on_error(error)

    def subscribe(self, *subscribers: Subscriber):
        self._subscribers += subscribers

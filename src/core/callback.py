import random
from abc import ABC
from typing import Iterable


class Callback(ABC):
    def on_begin(self):
        pass

    def on_pre_update(self, iteration: int):
        pass

    def on_update(self, iteration: int):
        pass

    def on_post_update(self, iteration: int):
        pass

    def on_end(self):
        pass

    def on_early_stop(self, iteration: int):
        pass

    def never(self):
        return Never(self)

    def timer(self, due_time: int = 0, period: int = 1):
        return Timer(self, due_time, period)

    def range(self, start: int, count: int):
        return Range(self, start, count)

    def random(self, prob: float):
        return Random(self, prob)


class CallbackWrapper(Callback):
    def __init__(self, callback: Callback | Iterable[Callback]):
        super().__init__()
        self.callbacks = [callback] if isinstance(callback, Callback) else callback
        self.active = True

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_pre_update(self, iteration: int):
        if self.active:
            for callback in self.callbacks:
                callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        if self.active:
            for callback in self.callbacks:
                callback.on_update(iteration)

    def on_post_update(self, iteration: int):
        if self.active:
            for callback in self.callbacks:
                callback.on_post_update(iteration)

    def on_end(self):
        for callback in self.callbacks:
            callback.on_end()

    def on_early_stop(self, iteration: int):
        for callback in self.callbacks:
            callback.on_early_stop(iteration)


class Never(CallbackWrapper):
    def __init__(self, callback: Callback):
        super().__init__(callback)

    def on_pre_update(self, iteration: int):
        self.active = False


class Timer(CallbackWrapper):
    def __init__(self, callback: Callback, due_time: int = 0, period: int = 1):
        super().__init__(callback)
        self._due_time = due_time
        self._period = period
        self._cnt = 0

    def on_pre_update(self, iteration: int):
        diff = self._cnt - self._due_time
        self.active = diff >= 0 and diff % self._period == 0
        self._cnt += 1
        super().on_pre_update(iteration)


class Range(CallbackWrapper):
    def __init__(self, callback: Callback, start: int, count: int):
        super().__init__(callback)
        self._start = start
        self._count = count

        self._cnt = 0

    def on_pre_update(self, iteration: int):
        self.active = self._cnt >= self._start and self._cnt < self._start + self._count
        self._cnt += 1
        super().on_pre_update(iteration)


class Random(CallbackWrapper):
    def __init__(self, callback: Callback, prob: float):
        super().__init__(callback)
        self._prob = prob

    def on_pre_update(self, iteration: int):
        self.active = random.random() < self._prob


class InOrder(CallbackWrapper):
    def __init__(
        self, callbacks: list[Callback], intervals: list[int], loop: bool = True
    ):
        assert len(callbacks) == len(
            intervals
        ), "The length of callbacks and intervals must be the same."

        super().__init__(callbacks)
        self._intervals = intervals
        self._loop = loop

        self._len = len(callbacks)
        self._cnt = 0
        self._idx = 0

    def on_pre_update(self, iteration: int):
        if self._cnt < self._intervals[self._idx]:
            self._cnt += 1
        else:
            self._idx += 1
            self._cnt = 1

        if self._loop and self._idx >= self._len:
            self._idx = 0

        if self._idx < self._len:
            self.callbacks[self._idx].on_pre_update(iteration)

    def on_update(self, iteration: int):
        if self._idx < self._len:
            self.callbacks[self._idx].on_update(iteration)

    def on_post_update(self, iteration: int):
        if self._idx < self._len:
            self.callbacks[self._idx].on_post_update(iteration)


class Shuffle(CallbackWrapper):
    def __init__(self, callbacks: list[Callback]):
        super().__init__(callbacks)
        self._order = list(range(len(self.callbacks)))
        self._idx = 0
        random.shuffle(self._order)

    def on_pre_update(self, iteration: int):
        if self._idx == len(self._order):
            random.shuffle(self._order)
            self._idx = 0

        self.callbacks[self._order[self._idx]].on_pre_update(iteration)

    def on_update(self, iteration: int):
        self.callbacks[self._order[self._idx]].on_update(iteration)

    def on_post_update(self, iteration: int):
        self.callbacks[self._order[self._idx]].on_post_update(iteration)
        self._idx += 1

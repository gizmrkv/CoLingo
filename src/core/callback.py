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

    def count(self):
        return Count(self)


class CallbackWrapper(Callback):
    def __init__(self, callback: Callback | Iterable[Callback]):
        super().__init__()
        self.callbacks = [callback] if isinstance(callback, Callback) else callback

    def on_begin(self):
        for callback in self.callbacks:
            callback.on_begin()

    def on_pre_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_update(iteration)

    def on_post_update(self, iteration: int):
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
        pass

    def on_update(self, iteration: int):
        pass

    def on_post_update(self, iteration: int):
        pass


class Timer(CallbackWrapper):
    def __init__(self, callback: Callback, due_time: int = 0, period: int = 1):
        super().__init__(callback)
        self._due_time = due_time
        self._period = period

        self._run = False
        self._cnt = 0

    def on_pre_update(self, iteration: int):
        diff = self._cnt - self._due_time
        self._run = diff >= 0 and diff % self._period == 0
        self._cnt += 1
        if self._run:
            for callback in self.callbacks:
                callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_update(iteration)

    def on_post_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_post_update(iteration)


class Range(CallbackWrapper):
    def __init__(self, callback: Callback, start: int, count: int):
        super().__init__(callback)
        self._start = start
        self._count = count

        self._cnt = 0

    def on_pre_update(self, iteration: int):
        self._run = self._cnt >= self._start and self._cnt < self._start + self._count
        self._cnt += 1
        if self._run:
            for callback in self.callbacks:
                callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_update(iteration)

    def on_post_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_post_update(iteration)


class Random(CallbackWrapper):
    def __init__(self, callback: Callback, prob: float):
        super().__init__(callback)
        self._prob = prob

        self._run = False

    def on_pre_update(self, iteration: int):
        self._run = random.random() < self._prob
        if self._run:
            for callback in self.callbacks:
                callback.on_pre_update(iteration)

    def on_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_update(iteration)

    def on_post_update(self, iteration: int):
        if self._run:
            for callback in self.callbacks:
                callback.on_post_update(iteration)


class Count(CallbackWrapper):
    def __init__(self, callback: Callback):
        super().__init__(callback)
        self._cnt = 0

    def on_pre_update(self, iteration: int):
        self._cnt += 1
        for callback in self.callbacks:
            callback.on_pre_update(self._cnt)

    def on_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_update(self._cnt)

    def on_post_update(self, iteration: int):
        for callback in self.callbacks:
            callback.on_post_update(self._cnt)


class InOrder(CallbackWrapper):
    def __init__(
        self, callbacks: list[Callback], intervals: list[int], loop: bool = True
    ):
        assert len(callbacks) == len(
            intervals
        ), "The length of callbacks and intervals must be the same."

        super().__init__(callbacks)
        self.intervals = intervals
        self.loop = loop

        self._len = len(callbacks)
        self._cnt = 0
        self._idx = 0

    def on_pre_update(self, iteration: int):
        if self._cnt < self.intervals[self._idx]:
            self._cnt += 1
        else:
            self._idx += 1
            self._cnt = 1

        if self.loop and self._idx >= self._len:
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
        self.order = list(range(len(self.callbacks)))

    def on_pre_update(self, iteration: int):
        random.shuffle(self.order)
        for i in self.order:
            self.callbacks[i].on_pre_update(iteration)

    def on_update(self, iteration: int):
        for i in self.order:
            self.callbacks[i].on_update(iteration)

    def on_post_update(self, iteration: int):
        for i in self.order:
            self.callbacks[i].on_post_update(iteration)

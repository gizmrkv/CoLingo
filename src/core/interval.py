from .callback import Callback


class Interval(Callback):
    def __init__(self, *callback: Callback, period: int, offset: int = 0):
        super().__init__()
        self._callback = callback
        self._period = period
        self._offset = offset
        self._count = 0

    def on_update(self, step: int):
        diff = self._count - self._offset
        if diff >= 0 and diff % self._period == 0:
            for callback in self._callback:
                callback.on_update()
        self._count += 1

    def on_begin(self):
        for callback in self._callback:
            callback.on_begin()

    def on_early_stop(self, step: int):
        for callback in self._callback:
            callback.on_early_stop()

    def on_end(self):
        for callback in self._callback:
            callback.on_end()

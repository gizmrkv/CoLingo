from ..core.callback import Callback


class Logger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = {}

    def log(self, logs: dict, flush: bool = False):
        self.logs |= logs
        if flush:
            self.flush()

    def flush(self):
        pass

    def on_post_update(self, iteration: int):
        self.flush()

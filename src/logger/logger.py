from ..callback import Callback


class Logger(Callback):
    def __init__(self):
        super().__init__()

    def log(self, *args, **kwargs):
        pass

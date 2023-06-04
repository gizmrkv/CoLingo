from pprint import pprint

from .logger import Logger


class ConsoleLogger(Logger):
    def flush(self):
        pprint(self.logs)
        self.logs = {}

from pprint import pprint


class Logger:
    def log(self, logs: dict):
        pass


class ConsoleLogger(Logger):
    def log(self, logs: dict):
        pprint(logs)

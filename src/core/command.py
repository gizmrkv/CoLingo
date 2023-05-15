from enum import Enum


class Command(Enum):
    SEND = 1
    RECEIVE = 2
    PREDICT = 3
    INTERPRET = 4
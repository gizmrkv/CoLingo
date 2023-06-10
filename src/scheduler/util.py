import math
import random
from itertools import count
from typing import Callable


def lambda_scheduler(f: Callable[[int], float], randomly: bool = False):
    pool = 0
    for i in count():
        if randomly:
            n = int(random.random() < f(i))
        else:
            pool, n = math.modf(pool + f(i))
            n = int(n)

        yield n

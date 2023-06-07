from typing import Callable

import numpy as np
from numba import njit


@njit
def pdist(X: np.ndarray, dist: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    n = X.shape[0]
    size = (n * (n - 1)) // 2
    distances = list(range(size))
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            distances[k] = dist(X[i], X[j])
            k += 1
    return distances


@njit
def edit_distance(s1: np.ndarray, s2: np.ndarray):
    """
    This is slower than the editdistance.eval, but it is compatible with numba.
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = np.arange(len(s2) + 1, dtype=np.int64)
    for i, c1 in enumerate(s1):
        current_row = np.zeros(len(s2) + 1, dtype=np.int64)
        current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]

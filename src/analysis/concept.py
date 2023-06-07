from typing import Tuple

import numpy as np
import torch as th
from numba import njit


@njit
def concept_distance(concept1: np.ndarray, concept2: np.ndarray):
    return np.mean(concept1 != concept2)


def concept_accuracy(
    input: th.Tensor,
    target: th.Tensor,
) -> Tuple[float, float, th.Tensor]:
    mark = input == target
    complete = mark.all(dim=-1).float().mean().item()
    attrs = mark.float().mean(dim=0)
    partial = attrs.mean().item()
    return partial, complete, attrs

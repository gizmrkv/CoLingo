from typing import Callable

import numpy as np
import torch as th
from numba import njit
from rapidfuzz.distance import (
    OSA,
    DamerauLevenshtein,
    Hamming,
    Indel,
    Jaro,
    JaroWinkler,
    LCSseq,
    Levenshtein,
    Postfix,
    Prefix,
)
from rapidfuzz.process import cdist
from scipy.stats import kendalltau, pearsonr, spearmanr

from .distance import norm_edit, pdist


def topographic_similarity(
    x: np.ndarray,
    y: np.ndarray,
    x_dist: str = "Hamming",
    y_dist: str = "Levenshtein",
    x_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    y_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    corr: str = "spearman",
    normalized: bool = False,
    workers: int = 1,
) -> float:
    dists = {
        "DamerauLevenshtein": DamerauLevenshtein,
        "Levenshtein": Levenshtein,
        "Hamming": Hamming,
        "Indel": Indel,
        "Jaro": Jaro,
        "JaroWinkler": JaroWinkler,
        "LCSseq": LCSseq,
        "OSA": OSA,
        "Postfix": Postfix,
        "Prefix": Prefix,
    }
    x_dist = dists[x_dist]
    y_dist = dists[y_dist]

    if normalized:
        x_dist = x_dist.normalized_distance
        y_dist = y_dist.normalized_distance
    else:
        x_dist = x_dist.distance
        y_dist = y_dist.distance

    x_dmat = cdist(x, x, scorer=x_dist, processor=x_processor, workers=workers)
    y_dmat = cdist(y, y, scorer=y_dist, processor=y_processor, workers=workers)

    x_pdist = x_dmat[np.triu_indices(n=x_dmat.shape[0], k=1)]
    y_pdist = y_dmat[np.triu_indices(n=y_dmat.shape[0], k=1)]

    corrs = {
        "spearman": spearmanr,
        "kendall": kendalltau,
        "pearson": pearsonr,
    }
    corr = corrs[corr]
    return corr(x_pdist, y_pdist).correlation


@njit
def levenshtein_language_similarity(
    x: np.ndarray, y: np.ndarray, x_len: np.ndarray, y_len: np.ndarray
) -> float:
    dist = np.arange(x.shape[0])
    for i in range(x.shape[0]):
        dist[i] = norm_edit(x[i, : x_len[i]], y[i, : y_len[i]])
    return 1 - dist.mean()

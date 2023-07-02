from typing import Callable

import numpy as np
import torch as th
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

str2distance = {
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


def topographic_similarity(
    x: np.ndarray,
    y: np.ndarray,
    x_dist: str = "Hamming",
    y_dist: str = "Levenshtein",
    corr: str = "spearman",
    x_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    y_processor: Callable[[np.ndarray], np.ndarray] | None = None,
    normalized: bool = True,
    workers: int = 1,
) -> float:
    x_dist = str2distance[x_dist]
    y_dist = str2distance[y_dist]

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


def language_similarity(
    x: np.ndarray,
    y: np.ndarray,
    dist: str,
    processor: Callable[[np.ndarray], np.ndarray] | None = None,
    normalized: bool = True,
) -> float:
    dist = str2distance[dist]
    if normalized:
        dist = dist.normalized_similarity
    else:
        dist = dist.similarity

    mean_dist = 0
    for xx, yy in zip(x, y):
        mean_dist += dist(xx, yy, processor=processor)

    return mean_dist / len(x)

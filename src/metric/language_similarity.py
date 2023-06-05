import numpy as np
import torch as th

from ..analysis import language_similarity
from .metric import Metric


class LanguageSimilarityMetric(Metric):
    def __init__(self, name: str = "lansim"):
        super().__init__(name=name)

    def calculate(
        self,
        languages: dict[str, np.ndarray],
        lengths: dict[str, np.ndarray],
        *args,
        **kwds,
    ):
        langs = list(languages.keys())
        n_langs = len(langs)
        lansims = {}
        for i in range(n_langs):
            for j in range(i + 1, n_langs):
                lang1 = languages[langs[i]]
                lang2 = languages[langs[j]]
                leng1 = lengths[langs[i]]
                leng2 = lengths[langs[j]]
                lansim = language_similarity(lang1, lang2, leng1, leng2)
                lansims[f"{langs[i]}-{langs[j]}"] = lansim
        lansims["mean"] = sum(lansims.values()) / len(lansims)
        return {self.name: lansims}

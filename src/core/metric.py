from typing import Any

import editdistance
import numpy as np
import torch as th
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from .util import concept_distance, message_similarity


class MessageLength:
    def __call__(self, aux_s, *args, **kwds):
        return aux_s["length"].float().mean().item()


class MessageEntropy:
    def __call__(self, aux_s, *args, **kwds):
        return aux_s["entropy"].mean().item()


class ConceptAccuracy:
    def __init__(self, n_attributes: int, n_values: int):
        self.n_attributes = n_attributes
        self.n_values = n_values

    def __call__(self, input: th.Tensor, target: th.Tensor, *args, **kwargs):
        batch_size = target.shape[0]
        input = (
            input.view(batch_size * self.n_attributes, -1)
            .argmax(dim=-1)
            .reshape(-1, self.n_attributes)
        )
        acc = (input == target).float().mean().item()
        return acc


class TopographicSimilarity:
    def __call__(
        self, concept: th.Tensor, languages: dict[str, th.Tensor], *args, **kwds
    ):
        topsims = {}
        concept_pdist = pdist(concept.detach().cpu().numpy(), concept_distance)
        for agent_name, language in languages.items():
            language_pdist = pdist(language.detach().cpu().numpy(), editdistance.eval)
            topsims[agent_name] = spearmanr(language_pdist, concept_pdist).correlation

        return topsims


class LanguageSimilarity:
    def __call__(self, languages: dict[str, th.Tensor], *args, **kwds):
        langs = list(languages.values())
        n_langs = len(langs)
        langsim = 0
        for i in range(n_langs):
            for j in range(i + 1, n_langs):
                langsim += message_similarity(langs[i], langs[j]).mean().item()

        return langsim / n_langs * (n_langs - 1) / 2

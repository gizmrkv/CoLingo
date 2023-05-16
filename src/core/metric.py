import editdistance
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


class UniqueMessage:
    def __call__(self, message: th.Tensor, *args, **kwds):
        n = message.shape[0]
        message = message.sort(dim=1)[0]
        message = th.unique(message, dim=0)
        return message.shape[0] / n


class SignalingDisplay:
    def __call__(
        self, input: th.Tensor, message: th.Tensor, target: th.Tensor, *args, **kwds
    ):
        bsz = input.shape[0]
        input = input.view(bsz * 2, 5).argmax(dim=-1).view(bsz, 2)
        for t, m, i in zip(target, message, input):
            print(f"{tuple(t.tolist())} -> {m.tolist()} -> {tuple(i.tolist())}")


class ConceptAccuracy:
    def __init__(self, n_attributes: int, n_values: int):
        self.n_attributes = n_attributes
        self.n_values = n_values

    def __call__(self, input: th.Tensor, target: th.Tensor, *args, **kwargs):
        input = input.argmax(dim=-1)
        acc = {}
        acc["partial"] = (input == target).float().mean().item()
        acc["complete"] = (input == target).all(dim=-1).float().mean().item()
        for i in range(self.n_attributes):
            acc[i + 1] = (input[:, i] == target[:, i]).float().mean().item()

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

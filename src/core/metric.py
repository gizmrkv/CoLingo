from abc import ABC, abstractmethod

import torch as th

from .util import language_similarity, topographic_similarity


class Metric(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def calculate(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class MessageMetric(Metric):
    def __init__(self, name: str = "message"):
        super().__init__(name=name)

    def calculate(
        self,
        message: th.Tensor,
        log_prob: th.Tensor,
        entropy: th.Tensor,
        length: th.Tensor,
        *args,
        **kwds,
    ):
        n = message.shape[0]
        message = message.sort(dim=1)[0]
        message = th.unique(message, dim=0)
        uniques = message.shape[0] / n
        return {
            self.name: {
                "log_prob": log_prob.mean().item(),
                "entropy": entropy.mean().item(),
                "length": length.float().mean().item(),
                "uniques": uniques,
            }
        }


# class SignalingDisplay:
#     def __call__(
#         self, input: th.Tensor, message: th.Tensor, output: th.Tensor, *args, **kwds
#     ):
#         bsz = input.shape[0]
#         input = input.view(bsz * 2, 5).argmax(dim=-1).view(bsz, 2)
#         for inp, mes, out in zip(input, message, output):
#             print(f"{tuple(inp.tolist())} -> {mes.tolist()} -> {tuple(out.tolist())}")


class ConceptAccuracyMetric(Metric):
    def __init__(
        self, n_attributes: int, n_values: int, name: str = "concept_accuracy"
    ):
        super().__init__(name=name)
        self.n_attributes = n_attributes
        self.n_values = n_values

    def calculate(self, output: th.Tensor, target: th.Tensor, *args, **kwargs):
        output = output.argmax(dim=-1)
        acc = {}
        acc["partial"] = (output == target).float().mean().item()
        acc["complete"] = (output == target).all(dim=-1).float().mean().item()
        for i in range(self.n_attributes):
            acc[i + 1] = (output[:, i] == target[:, i]).float().mean().item()

        return {self.name: acc}


class TopographicSimilarityMetric(Metric):
    def __init__(self, name: str = "topsim"):
        super().__init__(name=name)

    def calculate(
        self, input: th.Tensor, languages: dict[str, th.Tensor], *args, **kwds
    ):
        topsims = {}
        for agent_name, language in languages.items():
            topsims[agent_name] = topographic_similarity(
                input.numpy(), language.numpy()
            )

        return {self.name: topsims}


class LanguageSimilarityMetric(Metric):
    def __init__(self, name: str = "lansim"):
        super().__init__(name=name)

    def calculate(
        self,
        languages: dict[str, th.Tensor],
        lengths: dict[str, th.Tensor],
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

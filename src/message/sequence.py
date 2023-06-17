from dataclasses import dataclass

from torchtyping import TensorType


@dataclass
class SequenceMessage:
    batch_size: int
    max_len: int
    vocab_size: int
    sequence: TensorType["batch_size", "max_len", int]
    logits: TensorType["batch_size", "max_len", "vocab_size", float]
    log_probs: TensorType["batch_size", "max_len", float]
    log_prob: TensorType["batch_size", float]
    entropies: TensorType["batch_size", "max_len", float]
    entropy: TensorType["batch_size", float]
    length: TensorType["batch_size", int]

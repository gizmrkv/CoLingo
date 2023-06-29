from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..dataset import random_split
from ..model import DiscSeqRNNDecoder, DiscSeqRNNEncoder


@dataclass
class Config:
    # exp config
    n_epochs: int
    batch_size: int
    seed: int
    device: str

    # common config
    lr: float
    length: int
    n_values: int
    latent_dim: int

    # encoder config
    encoder_hidden_dim: int
    encoder_embed_dim: int
    encoder_rnn_type: str
    n_encoder_layers: int

    # decoder config
    decoder_hidden_dim: int
    decoder_embed_dim: int
    decoder_rnn_type: str
    n_decoder_layers: int


def run_disc_seq_rnn_exp(cfg: dict):
    cfg = Config(**cfg)

    encoder = DiscSeqRNNEncoder(
        n_values=cfg.n_values,
        output_dim=cfg.latent_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        embed_dim=cfg.encoder_embed_dim,
        rnn_type=cfg.encoder_rnn_type,
        n_layers=cfg.n_encoder_layers,
    )
    decoder = DiscSeqRNNDecoder(
        length=cfg.length,
        n_values=cfg.n_values,
        input_dim=cfg.latent_dim,
        hidden_dim=cfg.decoder_hidden_dim,
        embed_dim=cfg.decoder_embed_dim,
        rnn_type=cfg.decoder_rnn_type,
        n_layers=cfg.n_decoder_layers,
    )

    dataset = (
        torch.Tensor(list(product(torch.arange(cfg.n_values), repeat=cfg.length)))
        .long()
        .to(cfg.device)
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg.n_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            z = encoder(batch)
            _, logits = decoder(z, input=batch)
            logits = logits.view(-1, cfg.n_values)
            batch = batch.view(-1)
            loss = criterion(logits, batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            mean_acc = 0
            for batch in valid_dataloader:
                z = encoder(batch)
                x, _ = decoder(z, input=batch)
                x = x.view(-1)
                batch = batch.view(-1)
                acc = (x == batch).float().mean().item()
                mean_acc += acc

            print(f"Epoch {epoch} | Valid Acc: {mean_acc:.4f}")

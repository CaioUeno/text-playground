from typing import Callable, List, Optional, Tuple

import numpy as np
from torch import cat, device, nn, no_grad, tensor
from torch.cuda import is_available
from torch.utils.data import DataLoader
from tqdm import tqdm


class BiLSTMEmbedder(nn.Module):

    """ """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        word_embedding_dim: int,
        num_layers: int,
        padding_idx: int,
        sentence_embedding_dim: int,
    ):

        super(BiLSTMEmbedder, self).__init__()

        # configs
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.padding_idx = padding_idx
        self.sentence_embedding_dim = sentence_embedding_dim

        # LSTM configs
        self.num_layers = num_layers
        self.hidden_size = 64
        self.dropout = 0.2 if self.num_layers > 1 else 0

        self.device = device("cuda" if is_available() else "cpu")

        # layers
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.word_embedding_dim,
            padding_idx=self.padding_idx,
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            self.word_embedding_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout,
        )

        # Dense layer
        self.output = nn.Linear(
            self.max_length * 2,  # double neurons because bidirectional=True
            self.sentence_embedding_dim,
        )

    def forward(self, x):

        embeddings = self.embedding(x)

        # pack sequence to ignore padding token
        packed_x = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths=(x != self.padding_idx)
            .sum(1)
            .to("cpu"),  # length of each sentence in the batch must be in cpu
            batch_first=True,
            enforce_sorted=False,
        )

        lstm_output, (hidden_state, cell_state) = self.lstm(packed_x)

        # unpack (pad) sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_output, batch_first=True, total_length=self.max_length
        )

        # concat last hidden states of both directions [batch, seq_len, lstm_hidden_size * 2] (two last positions)
        hidden_concat = cat((lstm_output[:, :, -1], lstm_output[:, :, -2]), dim=1)

        return self.output(hidden_concat)


class EmbedderTrainer(nn.Module):

    """ """

    def __init__(self, embedder: nn.Module):

        super(EmbedderTrainer, self).__init__()

        self.embedder = embedder

    def forward(self, pairs):

        a = self.embedder(pairs[0])
        b = self.embedder(pairs[1])

        cos = nn.CosineSimilarity(dim=1)
        return cos(a, b)
